"""SAM2.1 image segmentation backend for Label Studio.

Architecture
------------
Checkpoints are downloaded at build time (download_models.py).
The model is loaded **lazily** via _ensure_model() on the first predict() call,
inside each gunicorn worker process after fork().  This avoids CUDA
initialisation in the master process ("Cannot re-initialize CUDA in forked
subprocess").  Do NOT use gunicorn --preload.

Model selection
--------------
MODEL_CONFIG and MODEL_CHECKPOINT environment variables specify the model.
Both must be set at container startup; the backend raises RuntimeError if either
is missing.  There is no built-in model dictionary.

SAM2 image API (facebookresearch/sam2)
--------------------------------------
  sam2_model = build_sam2(config_file, ckpt_path, device=DEVICE)
  predictor  = SAM2ImagePredictor(sam2_model)
  predictor.set_image(pil_image)   # encodes visual features
  masks, scores, logits = predictor.predict(
      point_coords  = np.array([[x, y], ...]),  # pixel coords
      point_labels  = np.array([1, 0, ...]),    # 1=foreground, 0=background
      box           = np.array([x0, y0, x1, y1]),
      multimask_output = True,
  )
  # masks:  bool [N, H, W]  (N=3 when multimask_output=True)
  # scores: float [N]
"""
from __future__ import annotations

import base64
import gc as _gc
import logging
import os
import tempfile
import threading
import time as _time_module
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import numpy as np
import requests
import torch
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/data/models"))
DEVICE = os.getenv("DEVICE", "cuda")
GPU_IDLE_TIMEOUT = int(os.getenv("GPU_IDLE_TIMEOUT_SECS", "3600"))

# ── Model config from environment ───────────────────────────────────────────────
# MODEL_CONFIG and MODEL_CHECKPOINT are the only supported way to specify a model.
# There is no fallback to a built-in model dictionary.
_ENV_CONFIG = os.getenv("MODEL_CONFIG", "").strip()
_ENV_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "").strip()

if not _ENV_CONFIG or not _ENV_CHECKPOINT:
    raise RuntimeError(
        "MODEL_CONFIG and MODEL_CHECKPOINT environment variables must both be set.  "
        "See docs/sam21-backend.md for configuration details."
    )


def _resolve_model_key() -> str:
    """Return the logical model key from the checkpoint filename for logging."""
    return Path(_ENV_CHECKPOINT).stem   # e.g. "sam2.1_hiera_large.pt" → "sam2.1_hiera_large"


def _get_config_and_ckpt() -> tuple[str, Path]:
    """Return (config_file, ckpt_path) from the environment.

    MODEL_CHECKPOINT may be a bare filename (resolved under MODEL_DIR)
    or an absolute path.
    """
    config_file = _ENV_CONFIG
    ckpt_name = _ENV_CHECKPOINT
    if os.path.isabs(ckpt_name):
        ckpt_path = Path(ckpt_name)
    else:
        ckpt_path = MODEL_DIR / ckpt_name
    return config_file, ckpt_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_internal_url(url: str) -> str:
    """Replace external Label Studio host with internal Docker service URL."""
    ls_internal = os.getenv("LABEL_STUDIO_URL", "").rstrip("/")
    if not ls_internal or not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return url
    internal = urlparse(ls_internal)
    if parsed.path.startswith(("/data/", "/api/", "/tasks/")):
        return urlunparse(parsed._replace(scheme=internal.scheme, netloc=internal.netloc))
    return url


def _load_image(url: str) -> Image.Image:
    """Load image from a local path or HTTP URL (supports Label Studio auth header)."""
    # Local file path — read directly
    if not url.startswith(("http://", "https://", "s3://", "gs://", "azure-blob://")):
        return Image.open(url).convert("RGB")
    # HTTP/HTTPS URL — download with auth
    api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    headers = {"Authorization": f"Token {api_key}"} if api_key else {}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(response.content)
        tmp_path = f.name
    img = Image.open(tmp_path).convert("RGB")
    os.unlink(tmp_path)
    return img


def _load_predictor(device: str):
    """Load SAM2ImagePredictor from MODEL_CONFIG / MODEL_CHECKPOINT env vars."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config_file, ckpt_path = _get_config_and_ckpt()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Run download_models.py or rebuild the Docker image."
        )
    model_key = _resolve_model_key()
    logger.info("Loading SAM2.1 model: %s (config=%s, ckpt=%s)", model_key, config_file, ckpt_path)

    sam2_model = build_sam2(config_file, str(ckpt_path), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    logger.info("SAM2.1 model loaded: %s on %s", model_key, device)
    return predictor


def _detect_autocast_dtype(device: str) -> Optional[torch.dtype]:
    """Choose autocast dtype based on GPU compute capability.

    Returns:
        torch.bfloat16  — Ampere (sm_80+): full TF32 + bfloat16 support
        torch.float16   — Turing/Volta (sm_70–79): fp16 is safe
        None            — Pascal or CPU: no autocast
    """
    if device == "cpu":
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        major = props.major
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return torch.bfloat16
        if major >= 7:
            return torch.float16
        return None
    except Exception:
        return None


def _mask_to_rle(mask: np.ndarray) -> list:
    """Convert boolean mask [H, W] to Label Studio RLE (list of ints).

    label_studio_converter expects pixel values in {0, 255}.
    SAM2 returns bool [H, W]; astype(uint8) gives {0, 1} so multiply by 255.
    """
    mask_u8 = mask.astype(np.uint8) * 255
    rle = brush.mask2rle(mask_u8)
    logger.debug(
        "_mask_to_rle: mask shape=%s nonzero=%d rle_len=%d rle_head=%s",
        mask.shape, int(mask.sum()), len(rle), rle[:10],
    )
    return rle



# ── Main Model Class ───────────────────────────────────────────────────────────

class NewModel(LabelStudioMLBase):
    """SAM2.1 image segmentation ML backend.

    Supports:
      - KeyPoint prompts (Object = foreground, Exclude = background)
      - RectangleLabels prompts (Object = foreground box, Exclude = background box)
      - Persistent model selection via Choices checkbox in labeling_config.xml
    """

    # Class-level state shared across requests in the same worker process
    _active_model_key: str = ""
    _predictor = None                    # SAM2ImagePredictor | None
    _autocast_dtype: Optional[torch.dtype] = None
    _lock: threading.Lock = threading.Lock()
    _last_activity: float = 0.0
    _idle_thread_started: bool = False

    def setup(self) -> None:
        """Called once per worker at startup (after gunicorn fork)."""
        self._start_idle_monitor()
        logger.info(
            "SAM2.1 image backend ready (model: %s, device: %s)",
            _resolve_model_key(), DEVICE,
        )

    # ── Model lifecycle ───────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Load the model (once per worker process, thread-safe)."""
        with self._lock:
            if self._predictor is not None:
                logger.debug("Using cached model: %s", self._active_model_key)
                return

            self._predictor = _load_predictor(DEVICE)
            self._active_model_key = _resolve_model_key()
            self._autocast_dtype = _detect_autocast_dtype(DEVICE)

    def _unload_model(self) -> None:
        """Release VRAM after idle timeout."""
        with self._lock:
            if self._predictor is None:
                return
            logger.info(
                "GPU idle timeout (%ds): unloading model %s",
                GPU_IDLE_TIMEOUT, self._active_model_key,
            )
            del self._predictor
            self._predictor = None
            self._active_model_key = ""
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _start_idle_monitor(self) -> None:
        """Start background thread to unload model after idle."""
        if NewModel._idle_thread_started or GPU_IDLE_TIMEOUT <= 0:
            return
        NewModel._idle_thread_started = True
        NewModel._last_activity = _time_module.monotonic()

        def _monitor():
            while True:
                _time_module.sleep(60)
                if NewModel._predictor is None:
                    continue
                idle = _time_module.monotonic() - NewModel._last_activity
                if idle >= GPU_IDLE_TIMEOUT:
                    self._unload_model()

        t = threading.Thread(target=_monitor, daemon=True, name="sam21-idle-monitor")
        t.start()

    # ── Model selection ───────────────────────────────────────────────────────
    # _resolve_model_key is a module-level function (see above class).

    # ── Prompt parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_prompts(
        context: Optional[dict],
        img_w: int,
        img_h: int,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract point_coords, point_labels, box from context results.

        Returns:
            point_coords: float32 [N, 2] in pixel coords, or None
            point_labels: int32  [N],    1=foreground / 0=background, or None
            box:          float32 [4],   [x0, y0, x1, y1] in pixel coords, or None
        """
        if not context:
            return None, None, None

        results = context.get("result", [])
        points_xy: list[list[float]] = []
        point_labels_list: list[int] = []
        boxes_fg: list[list[float]] = []
        boxes_bg: list[list[float]] = []

        for r in results:
            rtype = r.get("type", "")
            value = r.get("value", {})
            original_w = r.get("original_width", img_w)
            original_h = r.get("original_height", img_h)
            scale_x = original_w / 100.0
            scale_y = original_h / 100.0

            # KeyPoint prompts ──────────────────────────────────────────────
            if rtype == "keypointlabels":
                x_pct = value.get("x", 0)
                y_pct = value.get("y", 0)
                px = x_pct * scale_x
                py = y_pct * scale_y
                labels = value.get("keypointlabels", [])
                is_exclude = labels and labels[0] == "Exclude"
                points_xy.append([px, py])
                point_labels_list.append(0 if is_exclude else 1)

            # RectangleLabels prompts ───────────────────────────────────────
            elif rtype == "rectanglelabels":
                x_pct = value.get("x", 0)
                y_pct = value.get("y", 0)
                w_pct = value.get("width", 0)
                h_pct = value.get("height", 0)
                x0 = x_pct * scale_x
                y0 = y_pct * scale_y
                x1 = (x_pct + w_pct) * scale_x
                y1 = (y_pct + h_pct) * scale_y
                labels = value.get("rectanglelabels", [])
                is_exclude = labels and labels[0] == "Exclude"
                if is_exclude:
                    boxes_bg.append([x0, y0, x1, y1])
                else:
                    boxes_fg.append([x0, y0, x1, y1])

        # Convert to numpy ─────────────────────────────────────────────────
        point_coords: Optional[np.ndarray] = None
        point_labels: Optional[np.ndarray] = None
        if points_xy:
            point_coords = np.array(points_xy, dtype=np.float32)
            point_labels = np.array(point_labels_list, dtype=np.int32)

        # SAM2 supports one box prompt per call; use the first FG box.
        # BG boxes are converted to background keypoints (centre point).
        box: Optional[np.ndarray] = None
        if boxes_fg:
            box = np.array(boxes_fg[0], dtype=np.float32)
        for bg_box in boxes_bg:
            cx = (bg_box[0] + bg_box[2]) / 2
            cy = (bg_box[1] + bg_box[3]) / 2
            if point_coords is None:
                point_coords = np.array([[cx, cy]], dtype=np.float32)
                point_labels = np.array([0], dtype=np.int32)
            else:
                point_coords = np.vstack([point_coords, [cx, cy]])
                point_labels = np.append(point_labels, 0)

        return point_coords, point_labels, box

    @staticmethod
    def _extract_label_from_context_item(item: dict) -> Optional[str]:
        """Extract a label name from one context result item."""
        value = item.get("value", {})
        item_type = item.get("type", "")

        candidate_fields = [
            item_type,
            "brushlabels",
            "rectanglelabels",
            "keypointlabels",
            "labels",
        ]
        for field in candidate_fields:
            labels = value.get(field)
            if isinstance(labels, list) and labels:
                label = str(labels[0]).strip()
                if label:
                    return label
        return None

    def _resolve_brush_output(self, context: Optional[dict]) -> tuple[str, str, str]:
        """Resolve BrushLabels schema and output label dynamically.

        Priority:
          1) First non-Exclude label from context result
          2) First label from BrushLabels in parsed_label_config
          3) "Object" as last-resort compatibility fallback
        """
        from_name = "brush"
        to_name = "image"
        try:
            from_name, to_name, _ = self.get_first_tag_occurence("BrushLabels", "Image")
        except Exception as exc:
            logger.warning("get_first_tag_occurence(BrushLabels, Image) failed: %s", exc)

        brush_labels = self.parsed_label_config.get(from_name, {}).get("labels", [])
        allowed_labels = {str(label) for label in brush_labels}

        selected_label: Optional[str] = None
        if context:
            for item in context.get("result", []):
                item_to_name = item.get("to_name")
                if item_to_name and item_to_name != to_name:
                    continue

                label = self._extract_label_from_context_item(item)
                if not label:
                    continue

                lowered = label.lower()
                if lowered in {"exclude", "background"}:
                    continue

                if allowed_labels and label not in allowed_labels:
                    continue

                if label:
                    selected_label = label
                    break

        if selected_label is None:
            if brush_labels:
                selected_label = str(brush_labels[0])
            else:
                selected_label = "Object"

        return from_name, to_name, selected_label

    # ── predict() ────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: List[dict],
        context: Optional[dict] = None,
        **kwargs,
    ) -> ModelResponse:
        NewModel._last_activity = _time_module.monotonic()

        if not tasks:
            return ModelResponse(predictions=[])

        task = tasks[0]
        task_id = task.get("id")
        raw_image_url = task["data"].get("image", "")
        if not raw_image_url:
            logger.warning("No image URL in task")
            return ModelResponse(predictions=[])

        # Convert s3:// URLs to Label Studio resolve endpoint
        if raw_image_url.startswith("s3://"):
            ls_base = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
            fileuri = base64.b64encode(raw_image_url.encode()).decode()
            image_url = f"{ls_base}/tasks/{task_id}/resolve/?fileuri={fileuri}"
        else:
            image_url = _to_internal_url(raw_image_url)

        # Resolve and load model
        self._ensure_model()

        brush_from_name, brush_to_name, selected_label = self._resolve_brush_output(context)

        # Download image
        try:
            image = _load_image(image_url)
        except Exception as exc:
            logger.error("Failed to download image %s: %s", image_url, exc)
            return ModelResponse(predictions=[])

        img_w, img_h = image.size

        # Parse geometric prompts
        point_coords, point_labels, box = self._parse_prompts(context, img_w, img_h)

        if point_coords is None and box is None:
            logger.debug("No geometric prompts found — skipping inference")
            return ModelResponse(predictions=[])

        # Run inference
        try:
            results, scores_list = self._run_inference(
                image,
                point_coords,
                point_labels,
                box,
                img_w,
                img_h,
                brush_from_name,
                brush_to_name,
                selected_label,
            )
        except Exception as exc:
            logger.error("SAM2.1 inference error: %s", exc, exc_info=True)
            return ModelResponse(predictions=[])

        return ModelResponse(predictions=[{"result": results, "score": float(np.mean(scores_list)) if scores_list else 0.0}])

    def _run_inference(
        self,
        image: Image.Image,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
        img_w: int,
        img_h: int,
        from_name: str,
        to_name: str,
        selected_label: str,
    ) -> tuple[list[dict], list[float]]:
        """Run SAM2 prediction and convert output to Label Studio format."""
        predictor = self._predictor  # local ref for thread safety

        # Context manager for autocast (precision by GPU arch)
        dtype = self._autocast_dtype
        if dtype is not None and DEVICE != "cpu":
            ctx = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            import contextlib
            ctx = contextlib.nullcontext()

        with torch.inference_mode(), ctx:
            predictor.set_image(image)
            masks_np, scores_np, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

        # masks_np: [N, H, W] bool;  scores_np: [N] float
        # Return only the highest-scoring mask; keep all scores in TextArea for reference.
        scores_list: list[float] = [float(s) for s in scores_np]
        best_idx = int(np.argmax(scores_np))
        best_mask = masks_np[best_idx]
        best_score = scores_list[best_idx]

        rle = _mask_to_rle(best_mask)
        results: list[dict] = [{
            "id": str(uuid4())[:8],
            "type": "brushlabels",
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": [selected_label],
            },
            "to_name": to_name,
            "from_name": from_name,
            "image_rotation": 0,
            "original_width": img_w,
            "original_height": img_h,
            "score": best_score,
        }]

        # Append score summary to TextArea (all candidates for reference).
        # Fixed ID so Label Studio can update the existing result in-place on each
        # new prediction, rather than accumulating multiple TextArea submissions.
        if scores_list:
            score_lines = "\n".join(
                [f"model: {self._active_model_key}"]
                + [f"#{i+1}  score={s:.4f}{' ✓' if i == best_idx else ''}"
                   for i, s in enumerate(scores_list)]
            )
            results.append({
                "id": "sam21_scores",
                "type": "textarea",
                "value": {"text": [score_lines]},
                "to_name": to_name,
                "from_name": "scores",
            })

        logger.info(
            "SAM2.1 (%s): %d mask(s), scores=%s",
            self._active_model_key,
            len(masks_np),
            [f"{s:.3f}" for s in scores_list],
        )
        return results, scores_list
