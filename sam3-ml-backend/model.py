"""SAM3 interactive segmentation backend for Label Studio.

Implements LabelStudioMLBase with:
- Sam3TrackerModel (PVS — Promptable Visual Segmentation, SAM2 drop-in)
- Supports: keypoint (positive/negative) + rectangle box prompts
- Output: BrushLabels with Label Studio RLE encoding
- Image cache: PIL Image LRU to avoid repeated downloads
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from io import BytesIO
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
import requests
import torch
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
EMBED_CACHE_SIZE: int = int(os.getenv("EMBED_CACHE_SIZE", "50"))
EMBED_CACHE_TTL: float = float(os.getenv("EMBED_CACHE_TTL", "300"))

# Label names interpreted as negative (background) prompts
_NEGATIVE_LABELS = frozenset({"background", "negative", "neg", "背景"})


class SAM3Backend(LabelStudioMLBase):
    """SAM3 interactive segmentation backend.

    Workflow:
    1. User clicks on image in Label Studio → context.result = [keypointlabels/rectanglelabels]
    2. predict() parses the prompts, runs Sam3TrackerModel, returns BrushLabels RLE mask
    3. Label Studio displays the mask overlay; user can refine with more clicks

    Embedding optimization note:
        Currently caches PIL Images to avoid re-download (~100ms saved).
        Full image encoder re-runs on each click (~500–1500ms on GPU).
        Production optimization: cache encoder output via model.vision_encoder(pixel_values)
        and pass image_embeddings= on subsequent calls to the same task.
    """

    def setup(self) -> None:
        self.set("model_version", f"sam3-tracker:{MODEL_ID.split('/')[-1]}")

        hf_token: Optional[str] = os.getenv("HF_TOKEN") or None
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

        logger.info("Loading SAM3 model '%s' on %s (dtype=%s)", MODEL_ID, DEVICE, dtype)
        self._model = Sam3TrackerModel.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            token=hf_token,
        ).to(DEVICE).eval()
        self._processor = Sam3TrackerProcessor.from_pretrained(
            MODEL_ID,
            token=hf_token,
        )

        # PIL Image cache: {url_md5: (timestamp, PIL.Image)}
        self._image_cache: dict[str, tuple[float, Image.Image]] = {}

        logger.info("SAM3 model loaded. Device: %s", DEVICE)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: list[dict],
        context: Optional[dict] = None,
        **kwargs,
    ) -> ModelResponse:
        # Without an interactive prompt, return empty predictions.
        # SAM3 needs user input — it does not auto-segment.
        if not context or not context.get("result"):
            return ModelResponse(
                predictions=[{"result": [], "score": 0.0} for _ in tasks]
            )

        predictions = [self._predict_single(task, context) for task in tasks]
        return ModelResponse(predictions=predictions)

    def fit(self, event: str, data: dict, **kwargs) -> None:
        """Receives annotation events — fine-tuning not implemented."""
        logger.info("Received event '%s' (fine-tuning not implemented)", event)

    # ── Private: single task inference ───────────────────────────────────────

    def _predict_single(self, task: dict, context: dict) -> dict:
        image_url = self._get_image_url(task)
        if not image_url:
            logger.warning("No image URL in task data: %s", list(task.get("data", {}).keys()))
            return {"result": [], "score": 0.0}

        image = self._load_image(image_url)
        orig_w, orig_h = image.size

        points, labels, boxes = self._parse_context(context, orig_w, orig_h)
        if not points and not boxes:
            return {"result": [], "score": 0.0}

        # Build processor kwargs — points take priority over box when both present
        proc_kwargs: dict = {}
        if points:
            # Shape: [batch=1, num_obj=1, num_points, 2]
            proc_kwargs["input_points"] = [[[p for p in points]]]
            # Shape: [batch=1, num_obj=1, num_points]
            proc_kwargs["input_labels"] = [[labels]]
        elif boxes:
            # Use first box only; shape: [batch=1, num_obj=1, 4]
            proc_kwargs["input_boxes"] = [[[boxes[0]]]]

        inputs = self._processor(images=image, return_tensors="pt", **proc_kwargs)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        model_inputs = {
            k: v.to(dtype) if torch.is_floating_point(v) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model(**model_inputs, multimask_output=False)

        # post_process_masks returns list[Tensor[num_obj, num_masks, H, W]]
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu().float(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        mask_tensor = masks[0]  # first (only) image in batch
        # Shape guard: [num_obj, num_masks, H, W] or [num_obj, H, W]
        if mask_tensor.dim() == 4:
            mask_np = mask_tensor[0, 0].numpy().astype(bool)
        elif mask_tensor.dim() == 3:
            mask_np = mask_tensor[0].numpy().astype(bool)
        else:
            mask_np = mask_tensor.numpy().astype(bool)

        score = 0.9
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            try:
                score = float(outputs.iou_scores.cpu().flatten()[0])
            except Exception:
                pass

        brush_from_name, image_to_name, label_name = self._parse_label_config()

        # Convert mask → Label Studio RLE (NOT COCO RLE — use label_studio_converter)
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        rle = brush.mask2rle(mask_uint8)

        result = {
            "from_name": brush_from_name,
            "to_name": image_to_name,
            "type": "brushlabels",
            "original_width": orig_w,
            "original_height": orig_h,
            "image_rotation": 0,
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": [label_name],
            },
        }
        return {
            "result": [result],
            "score": score,
            "model_version": self.get("model_version"),
        }

    # ── Private: prompt parsing ───────────────────────────────────────────────

    def _parse_context(
        self,
        context: dict,
        orig_w: int,
        orig_h: int,
    ) -> tuple[list[list[float]], list[int], list[list[float]]]:
        """Extract pixel-coordinate points, SAM labels, and boxes from LS context.

        Label Studio sends percentage-based coordinates (0–100).
        SAM3 needs absolute pixel coordinates.
        """
        points: list[list[float]] = []
        labels: list[int] = []
        boxes: list[list[float]] = []

        for item in context.get("result", []):
            itype = item.get("type", "")
            value = item.get("value", {})
            iw = item.get("original_width", orig_w) or orig_w
            ih = item.get("original_height", orig_h) or orig_h

            if itype == "keypointlabels":
                x_px = value["x"] / 100.0 * iw
                y_px = value["y"] / 100.0 * ih
                kp_labels: list[str] = value.get("keypointlabels", ["Object"])
                # Map label name → SAM prompt polarity
                is_neg = any(k.lower() in _NEGATIVE_LABELS for k in kp_labels)
                points.append([x_px, y_px])
                labels.append(0 if is_neg else 1)

            elif itype == "rectanglelabels":
                x1 = value["x"] / 100.0 * iw
                y1 = value["y"] / 100.0 * ih
                x2 = x1 + value["width"] / 100.0 * iw
                y2 = y1 + value["height"] / 100.0 * ih
                boxes.append([x1, y1, x2, y2])

        return points, labels, boxes

    # ── Private: image loading ────────────────────────────────────────────────

    def _get_image_url(self, task: dict) -> Optional[str]:
        data = task.get("data", {})
        return data.get("image") or data.get("img") or data.get("url")

    def _load_image(self, url: str) -> Image.Image:
        """Download image with LRU + TTL cache. Handles LS-internal /data/ URLs."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        now = time.monotonic()

        # Evict expired entries
        self._image_cache = {
            k: v for k, v in self._image_cache.items()
            if now - v[0] < EMBED_CACHE_TTL
        }
        # Evict oldest when over size limit
        if len(self._image_cache) >= EMBED_CACHE_SIZE and cache_key not in self._image_cache:
            oldest = min(self._image_cache, key=lambda k: self._image_cache[k][0])
            del self._image_cache[oldest]

        if cache_key in self._image_cache:
            return self._image_cache[cache_key][1]

        image = self._fetch_image(url)
        self._image_cache[cache_key] = (now, image)
        return image

    def _fetch_image(self, url: str) -> Image.Image:
        ls_url = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
        api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
        headers = {"Authorization": f"Token {api_key}"} if api_key else {}

        # Resolve LS-internal relative paths
        if url.startswith("/data/") or url.startswith("/tasks/"):
            url = f"{ls_url}{url}"

        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    # ── Private: label config parsing ────────────────────────────────────────

    def _parse_label_config(self) -> tuple[str, str, str]:
        """Parse XML label config to extract BrushLabels and Image tag names.

        Returns (brush_from_name, image_to_name, first_label_value).
        Falls back to ("tag", "image", "Object") if parsing fails.
        """
        brush_from_name = "tag"
        image_to_name = "image"
        default_label = "Object"

        try:
            config = getattr(self, "label_config", None) or "<View/>"
            root = ET.fromstring(config)

            for elem in root.iter("BrushLabels"):
                brush_from_name = elem.get("name", brush_from_name)
                labels = [lbl.get("value", "Object") for lbl in elem.findall("Label")]
                if labels:
                    default_label = labels[0]
                break  # use first BrushLabels tag

            for elem in root.iter("Image"):
                image_to_name = elem.get("name", image_to_name)
                break

        except ET.ParseError as exc:
            logger.warning("Failed to parse label config: %s", exc)

        return brush_from_name, image_to_name, default_label
