"""SAM3 image segmentation backend for Label Studio.

Architecture
------------
Model is loaded at **module scope** (singleton) because label_studio_ml.api
creates a new MODEL_CLASS instance on every /predict request — keeping the
model in setup() would re-download / re-load on every call.

SAM3 image API (facebookresearch/sam3)
---------------------------------------
  processor = Sam3Processor(model, resolution=1008, device=DEVICE)
  state = processor.set_image(image)                # sets visual features
  state = processor.set_text_prompt(prompt, state)  # PCS text-only or mixed
  state = processor.add_geometric_prompt(           # box prompt (normalized cxcywh)
      box=[cx, cy, w, h], label=True/False, state=state)
  state["masks"]   → bool tensor [N, 1, H, W]
  state["scores"]  → float tensor [N]
  state["boxes"]   → float tensor [N, 4]  (pixel xyxy)

Three predict paths
--------------------
  1. Text-only  : set_text_prompt only
  2. Geo-only   : add_geometric_prompt only (model auto-adds 'visual' dummy text)
  3. Mixed      : set_text_prompt first, then add_geometric_prompt

SAM2 fallback
--------------
  When sam3 package is not installed, falls back to SAM2ImagePredictor which
  has the classic (masks, scores, logits) tuple interface. Text prompts are
  silently ignored with a WARNING in that mode.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List, Dict, Optional
from uuid import uuid4

import numpy as np
import torch
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1")
CHECKPOINT_FILENAME: str = os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.1_multiplex.pt")
MODEL_DIR: str = os.getenv("MODEL_DIR", "/data/models")

# PCS / text-prompt feature gate
ENABLE_PCS: bool = os.getenv("SAM3_ENABLE_PCS", "true").lower() == "true"
# Confidence threshold for text-prompt detections (Sam3Processor default = 0.5)
CONFIDENCE_THRESHOLD: float = float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5"))
# Return all detected masks from text-prompt (False = top-1 by score)
RETURN_ALL_MASKS: bool = os.getenv("SAM3_RETURN_ALL_MASKS", "false").lower() == "true"

# ── CUDA optimisations ─────────────────────────────────────────────────────────
if DEVICE == "cuda":
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# autocast context reused per inference call (module-scope context managers are
# not safe across fork — enter/exit is done inside _predict_sam3/_predict_sam2)
_autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16} if DEVICE == "cuda" else None

# ── Checkpoint download ────────────────────────────────────────────────────────
try:
    from huggingface_hub import hf_hub_download

    _hf_token: Optional[str] = os.getenv("HF_TOKEN") or None
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("Downloading SAM3 checkpoint '%s/%s' …", MODEL_ID, CHECKPOINT_FILENAME)
    _checkpoint_path: str = hf_hub_download(
        repo_id=MODEL_ID,
        filename=CHECKPOINT_FILENAME,
        local_dir=MODEL_DIR,
        token=_hf_token,
    )
    logger.info("Checkpoint cached at: %s", _checkpoint_path)
except Exception as _hf_err:
    raise RuntimeError(
        f"Failed to download SAM3 checkpoint from '{MODEL_ID}'. "
        "Ensure HF_TOKEN is set and the model license has been accepted at "
        f"https://huggingface.co/{MODEL_ID}"
    ) from _hf_err

# ── Model loading ──────────────────────────────────────────────────────────────
_USING_SAM2_FALLBACK: bool = False

try:
    from sam3.model_builder import build_sam3_image_model           # type: ignore[import]
    from sam3.model.sam3_image_processor import Sam3Processor       # type: ignore[import]

    logger.info("Loading SAM3 image model on %s …", DEVICE)
    _sam_model = build_sam3_image_model(
        checkpoint_path=_checkpoint_path,
        device=DEVICE,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    processor = Sam3Processor(
        _sam_model,
        resolution=1008,
        device=DEVICE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    logger.info("SAM3 image model loaded (PCS enabled=%s).", ENABLE_PCS)

except ImportError:
    _USING_SAM2_FALLBACK = True
    logger.warning(
        "sam3 package not found — falling back to SAM2ImagePredictor. "
        "Text prompts (PCS) will be IGNORED. "
        "Install facebookresearch/sam3 when available."
    )
    ROOT_DIR = os.getcwd()
    sys.path.insert(0, ROOT_DIR)
    from sam2.build_sam import build_sam2                           # type: ignore[import]
    from sam2.sam2_image_predictor import SAM2ImagePredictor        # type: ignore[import]

    _sam2_model = build_sam2(
        os.getenv("MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        _checkpoint_path,
        device=DEVICE,
    )
    processor = SAM2ImagePredictor(_sam2_model)  # type: ignore[assignment]
    logger.info("SAM2 image predictor loaded (SAM3 fallback).")


# ── Backend ────────────────────────────────────────────────────────────────────

class NewModel(LabelStudioMLBase):
    """SAM3 image segmentation backend for Label Studio.

    Label config must include:
      - <BrushLabels name="…" toName="…" smart="true"> — output masks
      - <Image name="…" value="$…">                    — input image
      - <TextArea name="…" toName="…">                 — text prompt (optional)
      - <KeyPointLabels name="…" toName="…" smart="true"> — point prompts (opt)
      - <RectangleLabels name="…" toName="…" smart="true"> — box prompts (opt)

    Predict paths (in order of priority):
      1. text + geometry  → mixed PCS + geometric grounding
      2. text only        → pure PCS (returns N detections)
      3. geometry only    → classic SAM2-style point/box segmentation
    """

    def setup(self) -> None:
        self.set("model_version", f"sam3-image:{MODEL_ID.split('/')[-1]}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        """Return BrushLabels prediction(s) for the given prompt context."""

        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")

        if not context or not context.get("result"):
            return ModelResponse(predictions=[])

        # ── Parse context ──────────────────────────────────────────────────────
        image_width  = context["result"][0]["original_width"]
        image_height = context["result"][0]["original_height"]

        text_prompt: Optional[str] = None
        point_coords: list[list[float]] = []
        point_labels: list[int] = []
        input_box: Optional[list[float]] = None
        selected_label: Optional[str] = None

        for ctx in context["result"]:
            ctx_type = ctx["type"]

            if ctx_type == "textarea":
                # TextArea value: {"text": ["user typed string"]}
                texts = ctx["value"].get("text", [])
                if texts:
                    candidate = str(texts[0]).strip()
                    if candidate:
                        text_prompt = candidate
                continue  # no x/y/label for textarea

            # All geometric types share x, y in percentage
            x = ctx["value"].get("x", 0.0) * image_width  / 100.0
            y = ctx["value"].get("y", 0.0) * image_height / 100.0

            label_list = ctx["value"].get(ctx_type, [])
            if label_list and selected_label is None:
                selected_label = label_list[0]

            if ctx_type == "keypointlabels":
                # is_positive: 1 = foreground, 0 = background (set by LS smart tool)
                point_labels.append(int(ctx.get("is_positive", 1)))
                point_coords.append([x, y])

            elif ctx_type == "rectanglelabels":
                box_w = ctx["value"].get("width",  0.0) * image_width  / 100.0
                box_h = ctx["value"].get("height", 0.0) * image_height / 100.0
                input_box = [x, y, x + box_w, y + box_h]  # pixel xyxy

        logger.debug(
            "text=%r  points=%s  labels=%s  box=%s  label=%s",
            text_prompt, point_coords, point_labels, input_box, selected_label,
        )

        has_text = ENABLE_PCS and text_prompt is not None and not _USING_SAM2_FALLBACK
        has_geo  = bool(point_coords) or input_box is not None

        if not has_text and not has_geo:
            if text_prompt and _USING_SAM2_FALLBACK:
                logger.warning(
                    "Text prompt '%s' ignored — SAM2 fallback does not support PCS.",
                    text_prompt,
                )
            return ModelResponse(predictions=[])

        # ── Load image ─────────────────────────────────────────────────────────
        img_url  = tasks[0]["data"][value]
        img_path = self.get_local_path(img_url, task_id=tasks[0].get("id"))
        image    = np.array(Image.open(img_path).convert("RGB"))

        # ── Run SAM3 predictor ─────────────────────────────────────────────────
        try:
            if _USING_SAM2_FALLBACK:
                return self._predict_sam2(
                    image, point_coords, point_labels, input_box,
                    selected_label, image_width, image_height,
                    from_name, to_name,
                )
            else:
                return self._predict_sam3(
                    image, text_prompt, point_coords, point_labels, input_box,
                    selected_label, image_width, image_height,
                    from_name, to_name,
                )
        except Exception as exc:
            logger.error("Predict failed: %s", exc, exc_info=True)
            return ModelResponse(predictions=[])

    def fit(self, event: str, data: dict, **kwargs) -> None:
        logger.info("Received event '%s' (fit not implemented)", event)

    # ── SAM3 path ──────────────────────────────────────────────────────────────

    def _predict_sam3(
        self,
        image: np.ndarray,
        text_prompt: Optional[str],
        point_coords: list,
        point_labels: list,
        input_box: Optional[list],
        selected_label: Optional[str],
        image_width: int,
        image_height: int,
        from_name: str,
        to_name: str,
    ) -> ModelResponse:
        """Run SAM3 Sam3Processor pipeline."""
        ctx = torch.autocast(**_autocast_kwargs) if _autocast_kwargs else None
        if ctx:
            ctx.__enter__()
        try:
            return self._predict_sam3_inner(
                image, text_prompt, point_coords, point_labels, input_box,
                selected_label, image_width, image_height, from_name, to_name,
            )
        finally:
            if ctx:
                ctx.__exit__(None, None, None)

    def _predict_sam3_inner(
        self,
        image: np.ndarray,
        text_prompt: Optional[str],
        point_coords: list,
        point_labels: list,
        input_box: Optional[list],
        selected_label: Optional[str],
        image_width: int,
        image_height: int,
        from_name: str,
        to_name: str,
    ) -> ModelResponse:
        state = processor.set_image(Image.fromarray(image))  # type: ignore[attr-defined]

        has_text = text_prompt is not None

        # Text prompt (PCS)
        if has_text:
            assert text_prompt is not None
            state = processor.set_text_prompt(prompt=text_prompt, state=state)  # type: ignore[attr-defined]

        # Geometric prompts
        # Sam3Processor.add_geometric_prompt() expects:
        #   box = [cx, cy, w, h]  normalized [0, 1]
        #   label = True (foreground) / False (background)
        if input_box is not None:
            x0, y0, x1, y1 = input_box
            cx = ((x0 + x1) / 2.0) / image_width
            cy = ((y0 + y1) / 2.0) / image_height
            w  = (x1 - x0) / image_width
            h  = (y1 - y0) / image_height
            state = processor.add_geometric_prompt(  # type: ignore[attr-defined]
                box=[cx, cy, w, h], label=True, state=state,
            )

        # Points: Sam3Processor has no add_point_prompt() — represent each point
        # as a tiny box (±0.5% of image dims) so foreground / background is preserved.
        for (px, py), lbl in zip(point_coords, point_labels):
            eps_x = 0.005
            eps_y = 0.005
            cx = px / image_width
            cy = py / image_height
            state = processor.add_geometric_prompt(  # type: ignore[attr-defined]
                box=[cx, cy, eps_x * 2, eps_y * 2],
                label=bool(lbl),
                state=state,
            )

        masks_tensor  = state.get("masks")   # [N, 1, H, W] bool
        scores_tensor = state.get("scores")  # [N] float

        if masks_tensor is None or masks_tensor.shape[0] == 0:
            logger.info("SAM3 returned no detections (threshold=%.2f).", CONFIDENCE_THRESHOLD)
            return ModelResponse(predictions=[])

        # Determine which masks to return
        n_masks = masks_tensor.shape[0]
        if RETURN_ALL_MASKS:
            indices = list(range(n_masks))
        else:
            best_idx = int(scores_tensor.argmax().item()) if scores_tensor is not None else 0
            indices = [best_idx]

        results = []
        for idx in indices:
            mask_np = masks_tensor[idx, 0].cpu().numpy().astype(np.uint8)
            score   = float(scores_tensor[idx]) if scores_tensor is not None else 1.0
            rle     = brush.mask2rle(mask_np * 255)
            results.append({
                "id":             str(uuid4())[:4],
                "from_name":      from_name,
                "to_name":        to_name,
                "type":           "brushlabels",
                "original_width":  image_width,
                "original_height": image_height,
                "image_rotation":  0,
                "value": {
                    "format":      "rle",
                    "rle":         rle,
                    "brushlabels": [selected_label] if selected_label else [],
                },
                "score":    score,
                "readonly": False,
            })

        best_score = float(scores_tensor[indices].max()) if scores_tensor is not None else 1.0
        return ModelResponse(predictions=[{
            "result":        results,
            "model_version": self.get("model_version"),
            "score":         best_score,
        }])

    # ── SAM2 fallback path ─────────────────────────────────────────────────────

    def _predict_sam2(
        self,
        image: np.ndarray,
        point_coords: list,
        point_labels: list,
        input_box: Optional[list],
        selected_label: Optional[str],
        image_width: int,
        image_height: int,
        from_name: str,
        to_name: str,
    ) -> ModelResponse:
        """Run SAM2 SAM2ImagePredictor pipeline (text prompts not supported)."""
        ctx = torch.autocast(**_autocast_kwargs) if _autocast_kwargs else None
        if ctx:
            ctx.__enter__()
        try:
            return self._predict_sam2_inner(
                image, point_coords, point_labels, input_box,
                selected_label, image_width, image_height, from_name, to_name,
            )
        finally:
            if ctx:
                ctx.__exit__(None, None, None)

    def _predict_sam2_inner(
        self,
        image: np.ndarray,
        point_coords: list,
        point_labels: list,
        input_box: Optional[list],
        selected_label: Optional[str],
        image_width: int,
        image_height: int,
        from_name: str,
        to_name: str,
    ) -> ModelResponse:
        processor.set_image(image)  # type: ignore[attr-defined]

        np_points = np.array(point_coords, dtype=np.float32) if point_coords else None
        np_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        np_box    = np.array(input_box,    dtype=np.float32) if input_box    else None

        masks, scores, _ = processor.predict(  # type: ignore[attr-defined]
            point_coords=np_points,
            point_labels=np_labels,
            box=np_box,
            multimask_output=True,
        )
        # Sort by score descending, take best
        best_idx  = int(np.argsort(scores)[::-1][0])
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])

        rle = brush.mask2rle(best_mask * 255)
        return ModelResponse(predictions=[{
            "result": [{
                "id":             str(uuid4())[:4],
                "from_name":      from_name,
                "to_name":        to_name,
                "type":           "brushlabels",
                "original_width":  image_width,
                "original_height": image_height,
                "image_rotation":  0,
                "value": {
                    "format":      "rle",
                    "rle":         rle,
                    "brushlabels": [selected_label] if selected_label else [],
                },
                "score":    best_score,
                "readonly": False,
            }],
            "model_version": self.get("model_version"),
            "score":         best_score,
        }])
