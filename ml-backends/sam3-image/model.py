"""SAM3 image segmentation backend for Label Studio.

Architecture
------------
Checkpoint is downloaded at module scope, but the model is loaded **lazily**
via _ensure_loaded() on the first predict() call — inside each gunicorn worker
process, after fork().  This avoids CUDA initialisation in the master process
which causes "Cannot re-initialize CUDA in forked subprocess" in all workers.
Do NOT use gunicorn --preload; it loads the app in the master before fork.

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

Three predict paths (all via Sam3Processor)
--------------------------------------------
  1. Text-only  : set_text_prompt only
  2. Geo-only   : add_geometric_prompt only
  3. Mixed      : set_text_prompt first, then add_geometric_prompt
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Dict, Optional
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import numpy as np
import torch
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _to_internal_url(url: str) -> str:
    """Replace external Label Studio host with internal Docker service URL.

    Task data stores the public URL (e.g. https://label-studio.example.com/…).
    The ML backend container must use the internal URL (LABEL_STUDIO_URL) to
    bypass Cloudflare Access / reverse-proxy authentication.
    """
    ls_internal = os.getenv("LABEL_STUDIO_URL", "").rstrip("/")
    if not ls_internal or not url:
        return url
    parsed = urlparse(url)
    internal = urlparse(ls_internal)
    # Only rewrite Label Studio paths — don't touch S3/MinIO/external URLs
    if parsed.path.startswith(("/data/", "/api/", "/tasks/")):
        return urlunparse(parsed._replace(scheme=internal.scheme, netloc=internal.netloc))
    return url


# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_IMAGE_MODEL_ID", os.getenv("SAM3_MODEL_ID", "facebook/sam3.1"))
CHECKPOINT_FILENAME: str = os.getenv("SAM3_IMAGE_CHECKPOINT_FILENAME", os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.pt"))
# PCS / text-prompt feature gate
ENABLE_PCS: bool = os.getenv("SAM3_ENABLE_PCS", "true").lower() == "true"
# Confidence threshold for text-prompt detections (Sam3Processor default = 0.5)
CONFIDENCE_THRESHOLD: float = float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5"))
# Return all detected masks from text-prompt (False = top-1 by score)
RETURN_ALL_MASKS: bool = os.getenv("SAM3_RETURN_ALL_MASKS", "false").lower() == "true"

# ── CUDA optimisations ─────────────────────────────────────────────────────────
# Deferred to _ensure_loaded() — after gunicorn fork — so CUDA is never
# initialised in the master process.  Do NOT call get_device_properties() here.
_autocast_kwargs: Optional[dict] = None  # set in _ensure_loaded after fork

# ── Checkpoint download ────────────────────────────────────────────────────────
def _download_with_progress(
    repo_id: str,
    filename: str,
    token: Optional[str],
) -> str:
    """Wrap hf_hub_download with a heartbeat thread that logs every 30 s.
    Checkpoint is stored in HF_HOME cache (hf-cache volume); path returned
    directly — no extra copy to a separate volume."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    # Suppress tqdm progress bars — they produce \r-heavy output in container logs.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    stop_event = threading.Event()

    def _log_progress() -> None:
        while not stop_event.is_set():
            logger.info("  … still downloading %s, please wait …", filename)
            stop_event.wait(30)

    monitor = threading.Thread(target=_log_progress, daemon=True)
    monitor.start()
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
    finally:
        stop_event.set()
    return path


try:
    _hf_token: Optional[str] = os.getenv("HF_TOKEN") or None

    logger.info("Downloading SAM3 checkpoint '%s/%s' …", MODEL_ID, CHECKPOINT_FILENAME)
    _checkpoint_path: str = _download_with_progress(
        repo_id=MODEL_ID,
        filename=CHECKPOINT_FILENAME,
        token=_hf_token,
    )
    logger.info("Checkpoint cached at: %s", _checkpoint_path)
except Exception as _hf_err:
    raise RuntimeError(
        f"Failed to download SAM3 checkpoint from '{MODEL_ID}'. "
        "Ensure HF_TOKEN is set and the model license has been accepted at "
        f"https://huggingface.co/{MODEL_ID}"
    ) from _hf_err

# ── Lazy model loading ─────────────────────────────────────────────────────────
# Model is NOT loaded at module scope. Loading here initialises CUDA in the
# gunicorn master process; after fork() every worker fails with:
#   RuntimeError: Cannot re-initialize CUDA in forked subprocess.
# _ensure_loaded() is called at the start of predict() — i.e. inside the worker
# process, after gunicorn has already forked. CUDA is then initialised cleanly
# in each worker.
_processor = None   # Sam3Processor
_init_lock = threading.Lock()


def _ensure_loaded() -> None:
    """Load SAM3 model on first call inside the worker process."""
    global _processor
    if _processor is not None:
        return
    with _init_lock:
        if _processor is not None:
            return

        # ── Reset CUDA state for forked workers ──────────────────────────
        # gunicorn --preload loads the Flask app in the master process.
        # If any transitive import initialises CUDA there, PyTorch marks
        # every forked worker as unsafe (_in_bad_fork = True).  Resetting
        # these flags allows the worker to initialise a fresh CUDA context.
        # Safe because CUDA contexts are per-process and not inherited
        # across fork — the worker must re-initialise from scratch anyway.
        import torch.cuda as _cuda
        _cuda._in_bad_fork = False
        _cuda._initialized = False

        # ── CUDA optimisations (safe here — after fork, before CUDA init) ──
        global _autocast_kwargs
        if DEVICE == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # TF32 only effective on Ampere (sm_80+); harmless on Volta (sm_70).
            # bfloat16 autocast valid from Volta (sm_70) onward.
            _autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16}

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
        _processor = Sam3Processor(
            _sam_model,
            resolution=1008,
            device=DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        logger.info("SAM3 image model loaded (PCS enabled=%s).", ENABLE_PCS)


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

        _ensure_loaded()  # CUDA init deferred to worker process (after gunicorn fork)

        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")

        if not context or not context.get("result"):
            return ModelResponse(predictions=[])

        # ── Parse context ──────────────────────────────────────────────────────
        # TextArea results don't carry original_width/height — find first geometric result.
        # When only TextArea is present (text-only path), dimensions are read from the
        # image itself after loading.
        geo_ctx = next(
            (r for r in context["result"] if r.get("type") != "textarea"),
            None,
        )
        image_width:  Optional[int] = geo_ctx["original_width"]  if geo_ctx else None
        image_height: Optional[int] = geo_ctx["original_height"] if geo_ctx else None

        text_prompt: Optional[str] = None
        point_coords: list[list[float]] = []
        point_labels: list[int] = []
        # Each entry: (box_xyxy_pixels, is_positive)
        # is_positive=False for "Exclude" label → negative exemplar in SAM3
        input_boxes: list[tuple[list[float], bool]] = []
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
            label_name = label_list[0] if label_list else ""
            # Only set selected_label from positive, non-background, non-empty labels
            _is_exclude = label_name.lower() == "exclude"
            if label_name and label_name.lower() not in ("background", "exclude") and selected_label is None:
                selected_label = label_name

            if ctx_type == "keypointlabels":
                # is_positive: LS sets this for smart tools; fall back to label name.
                is_pos = int(ctx.get(
                    "is_positive",
                    0 if label_name.lower() == "background" else 1,
                ))
                point_labels.append(is_pos)
                point_coords.append([x, y])

            elif ctx_type == "rectanglelabels":
                box_w = ctx["value"].get("width",  0.0) * image_width  / 100.0
                box_h = ctx["value"].get("height", 0.0) * image_height / 100.0
                input_boxes.append(([x, y, x + box_w, y + box_h], not _is_exclude))

        logger.debug(
            "text=%r  points=%s  labels=%s  boxes=%s  label=%s",
            text_prompt, point_coords, point_labels,
            [(b, "+" if p else "-") for b, p in input_boxes], selected_label,
        )

        has_text = ENABLE_PCS and text_prompt is not None
        has_geo  = bool(point_coords) or bool(input_boxes)

        if not has_text and not has_geo:
            return ModelResponse(predictions=[])

        # Text-only path: no geometric context → selected_label stays None.
        # Default to the first BrushLabels label so the mask gets a visible colour.
        if selected_label is None:
            brush_labels = self.parsed_label_config.get(from_name, {}).get("labels", [])
            selected_label = brush_labels[0] if brush_labels else "Object"

        # ── Load image ─────────────────────────────────────────────────────────
        img_url  = _to_internal_url(tasks[0]["data"][value])
        img_path = self.get_local_path(img_url, task_id=tasks[0].get("id"))
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            # Diagnose: log file size + first 64 bytes so we can tell whether
            # the download returned an image or an HTML/auth-error page.
            try:
                _sz = os.path.getsize(img_path)
                with open(img_path, "rb") as _f:
                    _hdr = _f.read(64)
                logger.error(
                    "Cannot open image '%s' (%d bytes). Header hex: %s  ASCII: %r",
                    img_path, _sz, _hdr.hex(), _hdr,
                )
            except Exception:
                pass
            raise

        # Text-only path: no geometric context, derive dimensions from the image.
        if image_width is None or image_height is None:
            image_width, image_height = pil_img.size  # PIL: (width, height)

        image = np.array(pil_img)

        # ── Run predictor ──────────────────────────────────────────────────────
        # All paths go through Sam3Processor (text, geo-only, text+geo).
        try:
            return self._predict_sam3(
                    image, text_prompt, point_coords, point_labels, input_boxes,
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
        input_boxes: list,          # list of (box_xyxy, is_positive)
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
                image, text_prompt, point_coords, point_labels, input_boxes,
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
        input_boxes: list,          # list of (box_xyxy, is_positive)
        selected_label: Optional[str],
        image_width: int,
        image_height: int,
        from_name: str,
        to_name: str,
    ) -> ModelResponse:
        state = _processor.set_image(Image.fromarray(image))  # type: ignore[attr-defined]

        has_text = text_prompt is not None

        # Text prompt (PCS)
        if has_text:
            assert text_prompt is not None
            state = _processor.set_text_prompt(prompt=text_prompt, state=state)  # type: ignore[attr-defined]

        # Geometric prompts — boxes (positive and negative/Exclude)
        # Sam3Processor.add_geometric_prompt() expects:
        #   box = [cx, cy, w, h]  normalized [0, 1]
        #   label = True (foreground) / False (background / Exclude)
        for box_xyxy, is_positive in input_boxes:
            x0, y0, x1, y1 = box_xyxy
            cx = ((x0 + x1) / 2.0) / image_width
            cy = ((y0 + y1) / 2.0) / image_height
            w  = (x1 - x0) / image_width
            h  = (y1 - y0) / image_height
            logger.debug("add_geometric_prompt box=[%.3f,%.3f,%.3f,%.3f] positive=%s", cx, cy, w, h, is_positive)
            state = _processor.add_geometric_prompt(  # type: ignore[attr-defined]
                box=[cx, cy, w, h], label=is_positive, state=state,
            )

        # Points: Sam3Processor has no add_point_prompt() — represent each point
        # as a tiny box (±0.5% of image dims) so foreground / background is preserved.
        for (px, py), lbl in zip(point_coords, point_labels):
            eps_x = 0.005
            eps_y = 0.005
            cx = px / image_width
            cy = py / image_height
            state = _processor.add_geometric_prompt(  # type: ignore[attr-defined]
                box=[cx, cy, eps_x * 2, eps_y * 2],
                label=bool(lbl),
                state=state,
            )

        masks_tensor  = state.get("masks")   # [N, 1, H, W] bool
        scores_tensor = state.get("scores")  # [N] float
        boxes_tensor  = state.get("boxes")   # [N, 4] float pixel xyxy (may be None)

        if masks_tensor is None or masks_tensor.shape[0] == 0:
            logger.info("SAM3 returned no detections (threshold=%.2f).", CONFIDENCE_THRESHOLD)
            return ModelResponse(predictions=[])

        n_total = masks_tensor.shape[0]
        # Log ALL detected candidates before filtering
        if scores_tensor is not None:
            for i in range(n_total):
                s = float(scores_tensor[i])
                b = boxes_tensor[i].cpu().tolist() if boxes_tensor is not None else None
                logger.info("  [SAM3] candidate %d  score=%.4f  box=%s", i, s, b)

        # Determine which masks to return
        if RETURN_ALL_MASKS:
            indices = list(range(n_total))
        else:
            best_idx = int(scores_tensor.argmax().item()) if scores_tensor is not None else 0
            indices = [best_idx]

        results = []
        score_lines: list[str] = []
        for rank, idx in enumerate(indices):
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
            box_str = ""
            if boxes_tensor is not None:
                x0, y0, x1, y1 = [int(v) for v in boxes_tensor[idx].cpu().tolist()]
                box_str = f"  box=[{x0},{y0},{x1},{y1}]"
            score_lines.append(f"#{rank}  score={score:.4f}{box_str}")

        # All candidates summary (even those filtered out)
        if scores_tensor is not None and n_total > len(indices):
            score_lines.append(f"(+{n_total - len(indices)} filtered candidates)")

        best_score = float(scores_tensor[indices].max()) if scores_tensor is not None else 1.0
        logger.info("[SAM3] returning %d mask(s), best=%.4f", len(indices), best_score)

        # Score display result — fills <TextArea name="scores"> in labeling config
        results.append({
            "id":        str(uuid4())[:4],
            "from_name": "scores",
            "to_name":   to_name,
            "type":      "textarea",
            "value":     {"text": ["\n".join(score_lines)]},
        })

        return ModelResponse(predictions=[{
            "result":        results,
            "model_version": self.get("model_version"),
            "score":         best_score,
        }])

