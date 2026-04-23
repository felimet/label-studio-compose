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

import base64
import hashlib
import logging
import os
import tempfile
import threading
import gc as _gc
import time as _time_module

import requests

try:
    from accelerate import dispatch_model, infer_auto_device_map  # type: ignore[import]
    _HAS_ACCELERATE: bool = True
except ImportError:
    _HAS_ACCELERATE: bool = False

from typing import Any, List, Dict, Optional
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
    # Only rewrite http/https LS URLs — s3://, gs://, etc. must pass through unchanged
    if parsed.scheme not in ("http", "https"):
        return url
    internal = urlparse(ls_internal)
    if parsed.path.startswith(("/data/", "/api/", "/tasks/")):
        return urlunparse(parsed._replace(scheme=internal.scheme, netloc=internal.netloc))
    return url


def _download_ls_url(url: str) -> str:
    """Download from a Label Studio internal URL directly with API token auth.

    Bypasses SDK get_local_path which mishandles resolve URLs: it decodes the
    S3 fileuri parameter and constructs an incorrect /data/<key> local path,
    causing 404 for S3-backed cloud storage in proxy mode.
    """
    api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    headers = {"Authorization": f"Token {api_key}"} if api_key else {}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_dir = os.path.join(tempfile.gettempdir(), "ls-ml-img-cache")
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, f"{url_hash}.cache")
    if not os.path.exists(filepath):
        with open(filepath, "wb") as _f:
            _f.write(r.content)
    return filepath


def _ls_api_get_json(path: str, timeout_sec: int = 10) -> Optional[Dict[str, Any]]:
    """GET Label Studio API JSON payload with Token auth.

    Used as a fallback for smart-trigger calls where context.result omits
    non-geometric controls and task payload doesn't include annotations.
    """

    ls_base = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
    api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    headers = {"Authorization": f"Token {api_key}"} if api_key else {}
    normalized_path = path if path.startswith("/") else f"/{path}"
    url = f"{ls_base}{normalized_path}"

    try:
        response = requests.get(url, headers=headers, timeout=timeout_sec)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.debug("Label Studio API GET failed (%s): %s", url, exc)
        return None

    if isinstance(payload, dict):
        return payload
    return None


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
# Candidate selection strategy when RETURN_ALL_MASKS is disabled.
# Supported: adaptive | top1 | topk | threshold
MASK_SELECTION_MODE: str = os.getenv("SAM3_MASK_SELECTION_MODE", "all").strip().lower()
VALID_SELECTION_MODES = {"adaptive", "top1", "topk", "threshold", "all"}
SELECTION_MODE_ALIASES = {
    "top-1": "top1",
    "return_all": "all",
    "all_masks": "all",
}
# Maximum masks returned for topk/adaptive modes.
MAX_RETURNED_MASKS: int = max(1, int(os.getenv("SAM3_MAX_RETURNED_MASKS", "3")))
# Minimum score gate for returned masks; 0 disables score filtering.
MIN_RETURN_SCORE: float = float(os.getenv("SAM3_MIN_RETURN_SCORE", "0.0"))
# If true, confidence threshold is applied to all selection modes.
# If false, confidence threshold is only enforced in threshold mode.
APPLY_THRESHOLD_GLOBALLY: bool = os.getenv("SAM3_APPLY_THRESHOLD_GLOBALLY", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Fallback when native point prompts are unavailable: half-size of point box (normalized).
# Final box size is (2 * SAM3_POINT_FALLBACK_HALF_SIZE) in each dimension.
POINT_FALLBACK_HALF_SIZE: float = float(os.getenv("SAM3_POINT_FALLBACK_HALF_SIZE", "0.005"))

# ── SAM3 Agent Configuration ───────────────────────────────────────────────────
AGENT_ENABLED: bool = os.getenv("SAM3_AGENT_ENABLED", "false").lower() == "true"
AGENT_LLM_URL: str = os.getenv("SAM3_AGENT_LLM_URL", "http://localhost:8000/v1/chat/completions")
AGENT_LLM_KEY: str = os.getenv("SAM3_AGENT_LLM_KEY", "")
AGENT_MODEL_NAME: str = os.getenv("SAM3_AGENT_MODEL_NAME", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")

# ── CUDA optimisations ─────────────────────────────────────────────────────────
# Deferred to _ensure_loaded() — after gunicorn fork — so CUDA is never
# initialised in the master process.  Do NOT call get_device_properties() here.
_autocast_kwargs: Optional[dict] = None  # set in _ensure_loaded after fork

# ── Checkpoint download ────────────────────────────────────────────────────────
def _setup_precision() -> Optional[dict]:
    """Detect GPU compute capability and configure autocast precision.

    ALL CUDA tiers use bfloat16 autocast.  This is a hard requirement imposed by
    sam3/perflib/fused.py::addmm_act, which unconditionally casts fc1 inputs and
    weights to bfloat16 (`.to(torch.bfloat16)`) before calling the fused kernel.
    Because fc1's output is always bf16, the downstream fc2 layer must also receive
    bf16 input — its weight must be cast to bf16.  torch.autocast(dtype=bfloat16)
    achieves this transparently for all standard linear/matmul ops at runtime.
    Without autocast, any fp32 weight in fc2 causes:
      RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float

    Performance notes per GPU generation:
      sm_80+ Ampere: native BF16 Tensor Core — fast path; TF32 enabled for any
        remaining fp32 ops (allow_tf32 flags).
      sm_70–79 Volta/Turing: no native BF16 hardware; bf16 ops are software-
        emulated (slower than fp32, but correctness trumps speed here).
      sm_61 and below Pascal: same software-emulation caveat as Volta.
        SAM3 was not designed for Pascal; performance will be poor.

    Multi-GPU note:
      torch.autocast is a global context — it cannot be configured per-device.
      min_major across all visible GPUs determines the tier.
      Override with TORCH_DTYPE=fp16 or TORCH_DTYPE=bf16 when needed.
    """
    # Allow manual override via environment variable
    _dtype_override = os.getenv("TORCH_DTYPE", "").lower()
    _dtype_map = {"fp16": torch.float16, "float16": torch.float16,
                  "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
    if _dtype_override in _dtype_map:
        logger.info("Precision: %s autocast — TORCH_DTYPE override", _dtype_override)
        return {"device_type": "cuda", "dtype": _dtype_map[_dtype_override]}

    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    if n == 0:
        return None
    min_major = min(torch.cuda.get_device_properties(i).major for i in range(n))
    if min_major >= 8:  # Ampere (sm_80+): BF16 Tensor Core native; TF32 for fp32 fallback ops
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Precision: bfloat16 autocast + TF32 — Ampere+ (min sm_%d0, %d GPU(s))", min_major, n)
        return {"device_type": "cuda", "dtype": torch.bfloat16}
    elif min_major >= 7:  # Volta/Turing (sm_70–79): bf16 software-emulated; no TF32
        logger.info("Precision: bfloat16 autocast — Volta/Turing (min sm_%d0, %d GPU(s))", min_major, n)
        return {"device_type": "cuda", "dtype": torch.bfloat16}
    else:  # Pascal (sm_60/61) and below: bfloat16 autocast (image model uses main branch, no addmm_act)
        logger.info("Precision: bfloat16 autocast — Pascal (min sm_%d0, %d GPU(s))", min_major, n)
        return {"device_type": "cuda", "dtype": torch.bfloat16}

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
_processor_runtime_lock = threading.Lock()
_runtime_controls_cache_lock = threading.Lock()
_runtime_controls_cache: Dict[str, Dict[str, Any]] = {}
_runtime_controls_cache_max_entries: int = int(os.getenv("SAM3_RUNTIME_CONTROLS_CACHE_SIZE", "2048"))

_last_used: float = 0.0
_IDLE_TIMEOUT: int = int(os.getenv("GPU_IDLE_TIMEOUT_SECS", "3600"))  # default: 1 hour


def _idle_watchdog() -> None:
    """Daemon: unloads model and frees VRAM after GPU_IDLE_TIMEOUT_SECS seconds idle.

    Checked every 60 s. Timer resets on each predict() call via _last_used update.
    """
    global _processor, _last_used
    while True:
        _time_module.sleep(60)
        if _processor is None:
            continue
        if _time_module.monotonic() - _last_used > _IDLE_TIMEOUT:
            with _init_lock:
                if _processor is not None and _time_module.monotonic() - _last_used > _IDLE_TIMEOUT:
                    logger.info("GPU idle >%ds — unloading model to free VRAM.", _IDLE_TIMEOUT)
                    _processor = None
                    if DEVICE.startswith("cuda"):
                        torch.cuda.empty_cache()
                        _gc.collect()


_watchdog_thread = threading.Thread(target=_idle_watchdog, daemon=True, name="gpu-idle-watchdog")
_watchdog_thread.start()


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

        # ── CUDA precision (safe here — after fork, before CUDA init) ─────────────
        # Precision: all CUDA GPUs → bfloat16 autocast (addmm_act constraint); Ampere+ also enables TF32.
        global _autocast_kwargs
        _autocast_kwargs = _setup_precision()

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
        # Multi-GPU: dispatch model layers across all available GPUs (device_map="auto").
        # Uses HuggingFace accelerate; falls back gracefully if not installed.
        # Each gunicorn worker process runs this independently after fork.
        if torch.cuda.device_count() > 1 and _HAS_ACCELERATE:
            _max_mem = {
                i: torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            }
            _dev_map = infer_auto_device_map(_sam_model, max_memory=_max_mem)
            _sam_model = dispatch_model(_sam_model, device_map=_dev_map)
            logger.info(
                "SAM3 image model dispatched across %d GPU(s) via device_map=auto.",
                torch.cuda.device_count(),
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
        - <TextArea name="confidence_threshold" toName="…"> — threshold override (optional)
                - <Choices name="selection_mode" toName="…">   — selection mode override (optional)
                    (legacy <TextArea name="selection_mode"> is still accepted)
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
        global _last_used
        _last_used = _time_module.monotonic()

        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")

        if not context or not context.get("result"):
            return ModelResponse(predictions=[])

        # ── Parse context ──────────────────────────────────────────────────────
        # Non-geometric controls (TextArea/Choices) don't carry original_width/height.
        # Find first result that includes image dimensions.
        # When no geometric result is present (text-only path), dimensions are read from
        # the image itself after loading.
        geo_ctx = next(
            (
                r
                for r in context["result"]
                if "original_width" in r and "original_height" in r
            ),
            None,
        )
        image_width:  Optional[int] = geo_ctx["original_width"]  if geo_ctx else None
        image_height: Optional[int] = geo_ctx["original_height"] if geo_ctx else None

        pure_text_prompt: Optional[str] = None
        mixed_text_prompt: Optional[str] = None
        confidence_threshold: float = CONFIDENCE_THRESHOLD
        selection_mode_override: Optional[str] = None
        apply_threshold_globally_override: Optional[bool] = None
        max_returned_masks: int = MAX_RETURNED_MASKS
        pure_text_prompt_source: str = "default"
        mixed_text_prompt_source: str = "default"
        confidence_threshold_source: str = "default"
        selection_mode_source: str = "default"
        apply_threshold_globally_source: str = "default"
        max_returned_masks_source: str = "default"
        point_coords: list[list[float]] = []
        point_labels: list[int] = []
        # Each entry: (box_xyxy_pixels, is_positive)
        # is_positive=False for "Exclude" label → negative exemplar in SAM3
        input_boxes: list[tuple[list[float], bool]] = []
        selected_label: Optional[str] = None

        for ctx in context["result"]:
            ctx_type = str(ctx.get("type", "") or "")

            if ctx_type == "choices":
                from_name_hint = str(ctx.get("from_name", "") or "")
                choices = ctx.get("value", {}).get("choices", [])
                if from_name_hint == "selection_mode" and choices:
                    normalized_mode = self._normalize_selection_mode(str(choices[0]))
                    if normalized_mode is None:
                        logger.warning(
                            "Invalid selection_mode choice '%s'; fallback to default mode",
                            choices[0],
                        )
                    else:
                        selection_mode_override = normalized_mode
                        selection_mode_source = "context"
                elif from_name_hint == "apply_threshold_globally":
                    if not choices:
                        apply_threshold_globally_override = False
                        apply_threshold_globally_source = "context"
                    else:
                        normalized_bool = self._normalize_boolean_value(str(choices[0]))
                        if normalized_bool is None:
                            logger.warning(
                                "Invalid apply_threshold_globally choice '%s'; fallback to env default %s",
                                choices[0],
                                APPLY_THRESHOLD_GLOBALLY,
                            )
                        else:
                            apply_threshold_globally_override = normalized_bool
                            apply_threshold_globally_source = "context"
                continue

            if ctx_type == "textarea":
                from_name_hint = str(ctx.get("from_name", "") or "")
                if from_name_hint == "scores":
                    continue

                # TextArea value: {"text": ["user typed string"]}
                texts = ctx.get("value", {}).get("text", [])
                if isinstance(texts, str):
                    texts = [texts]
                if texts:
                    candidate = str(texts[0]).strip()
                    if candidate:
                        if from_name_hint == "confidence_threshold":
                            try:
                                parsed_threshold = float(candidate)
                                if not np.isfinite(parsed_threshold):
                                    raise ValueError("non-finite")
                                clamped_threshold = float(np.clip(parsed_threshold, 0.0, 1.0))
                                if clamped_threshold != parsed_threshold:
                                    logger.warning(
                                        "confidence_threshold out of range (%.4f); clamped to %.4f",
                                        parsed_threshold,
                                        clamped_threshold,
                                    )
                                confidence_threshold = clamped_threshold
                                confidence_threshold_source = "context"
                            except (TypeError, ValueError):
                                logger.warning(
                                    "Invalid confidence_threshold input '%s'; fallback to env default %.4f",
                                    candidate,
                                    CONFIDENCE_THRESHOLD,
                                )
                        elif from_name_hint == "selection_mode":
                            normalized_mode = self._normalize_selection_mode(candidate)
                            if normalized_mode is None:
                                logger.warning(
                                    "Invalid selection_mode input '%s'; fallback to default mode",
                                    candidate,
                                )
                            else:
                                selection_mode_override = normalized_mode
                                selection_mode_source = "context"
                        elif from_name_hint == "apply_threshold_globally":
                            normalized_bool = self._normalize_boolean_value(candidate)
                            if normalized_bool is None:
                                logger.warning(
                                    "Invalid apply_threshold_globally input '%s'; fallback to env default %s",
                                    candidate,
                                    APPLY_THRESHOLD_GLOBALLY,
                                )
                            else:
                                apply_threshold_globally_override = normalized_bool
                                apply_threshold_globally_source = "context"
                        elif from_name_hint in ("selection_topk_k", "max_returned_masks"):
                            try:
                                parsed_k_raw = float(candidate)
                                if not np.isfinite(parsed_k_raw):
                                    raise ValueError("non-finite")
                                if not parsed_k_raw.is_integer():
                                    raise ValueError("non-integer")
                                parsed_k = int(parsed_k_raw)
                                if parsed_k < 1:
                                    logger.warning(
                                        "selection_topk_k must be >= 1 (got %d); clamped to 1",
                                        parsed_k,
                                    )
                                    parsed_k = 1
                                max_returned_masks = parsed_k
                                max_returned_masks_source = "context"
                            except (TypeError, ValueError):
                                logger.warning(
                                    "Invalid selection_topk_k input '%s'; fallback to env default %d",
                                    candidate,
                                    MAX_RETURNED_MASKS,
                                )
                        elif from_name_hint == "text_prompt_mixed":
                            mixed_text_prompt = candidate
                            mixed_text_prompt_source = "context"
                        elif from_name_hint in ("", "text_prompt"):
                            # Backward compatibility: keep supporting legacy text_prompt.
                            if pure_text_prompt is None:
                                pure_text_prompt = candidate
                                pure_text_prompt_source = "context"
                        elif pure_text_prompt is None:
                            # Backward compatibility for custom prompt field names.
                            pure_text_prompt = candidate
                            pure_text_prompt_source = "context"
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

        # Smart-trigger payloads from some Label Studio flows may omit non-geometric
        # control widgets in context.result. Recover control values from the matching
        # annotation result as a fallback, while keeping context values highest-priority.
        fallback_control_results = self._get_annotation_results_for_context(
            tasks[0] if tasks else {},
            context,
        )
        if fallback_control_results:
            for ctx in fallback_control_results:
                ctx_type = str(ctx.get("type", "") or "")

                if ctx_type == "choices":
                    from_name_hint = str(ctx.get("from_name", "") or "")
                    choices = ctx.get("value", {}).get("choices", [])
                    if from_name_hint == "selection_mode" and selection_mode_source == "default" and choices:
                        normalized_mode = self._normalize_selection_mode(str(choices[0]))
                        if normalized_mode is not None:
                            selection_mode_override = normalized_mode
                            selection_mode_source = "annotation"
                    elif from_name_hint == "apply_threshold_globally" and apply_threshold_globally_source == "default":
                        if not choices:
                            apply_threshold_globally_override = False
                            apply_threshold_globally_source = "annotation"
                        else:
                            normalized_bool = self._normalize_boolean_value(str(choices[0]))
                            if normalized_bool is not None:
                                apply_threshold_globally_override = normalized_bool
                                apply_threshold_globally_source = "annotation"
                    continue

                if ctx_type != "textarea":
                    continue

                from_name_hint = str(ctx.get("from_name", "") or "")
                if from_name_hint == "scores":
                    continue

                texts = ctx.get("value", {}).get("text", [])
                if isinstance(texts, str):
                    texts = [texts]
                if not texts:
                    continue

                candidate = str(texts[0]).strip()
                if not candidate:
                    continue

                if from_name_hint == "confidence_threshold" and confidence_threshold_source == "default":
                    try:
                        parsed_threshold = float(candidate)
                        if not np.isfinite(parsed_threshold):
                            raise ValueError("non-finite")
                        confidence_threshold = float(np.clip(parsed_threshold, 0.0, 1.0))
                        confidence_threshold_source = "annotation"
                    except (TypeError, ValueError):
                        pass
                elif from_name_hint == "selection_mode" and selection_mode_source == "default":
                    normalized_mode = self._normalize_selection_mode(candidate)
                    if normalized_mode is not None:
                        selection_mode_override = normalized_mode
                        selection_mode_source = "annotation"
                elif from_name_hint == "apply_threshold_globally" and apply_threshold_globally_source == "default":
                    normalized_bool = self._normalize_boolean_value(candidate)
                    if normalized_bool is not None:
                        apply_threshold_globally_override = normalized_bool
                        apply_threshold_globally_source = "annotation"
                elif from_name_hint in ("selection_topk_k", "max_returned_masks") and max_returned_masks_source == "default":
                    try:
                        parsed_k_raw = float(candidate)
                        if not np.isfinite(parsed_k_raw):
                            raise ValueError("non-finite")
                        if not parsed_k_raw.is_integer():
                            raise ValueError("non-integer")
                        max_returned_masks = max(1, int(parsed_k_raw))
                        max_returned_masks_source = "annotation"
                    except (TypeError, ValueError):
                        pass
                elif from_name_hint == "text_prompt_mixed" and mixed_text_prompt_source == "default":
                    mixed_text_prompt = candidate
                    mixed_text_prompt_source = "annotation"
                elif from_name_hint in ("", "text_prompt") and pure_text_prompt_source == "default":
                    pure_text_prompt = candidate
                    pure_text_prompt_source = "annotation"
                elif pure_text_prompt_source == "default":
                    pure_text_prompt = candidate
                    pure_text_prompt_source = "annotation"

        runtime_cache_key = self._runtime_controls_cache_key(
            tasks[0] if tasks else {},
            context,
        )
        cached_controls = self._get_cached_runtime_controls(runtime_cache_key) if runtime_cache_key else {}
        if cached_controls:
            if pure_text_prompt_source == "default" and cached_controls.get("pure_text_prompt"):
                pure_text_prompt = str(cached_controls["pure_text_prompt"]).strip() or None
                if pure_text_prompt:
                    pure_text_prompt_source = "cache"
            if mixed_text_prompt_source == "default" and cached_controls.get("mixed_text_prompt"):
                mixed_text_prompt = str(cached_controls["mixed_text_prompt"]).strip() or None
                if mixed_text_prompt:
                    mixed_text_prompt_source = "cache"
            if confidence_threshold_source == "default" and "confidence_threshold" in cached_controls:
                try:
                    confidence_threshold = float(np.clip(float(cached_controls["confidence_threshold"]), 0.0, 1.0))
                    confidence_threshold_source = "cache"
                except (TypeError, ValueError):
                    pass
            if selection_mode_source == "default" and "selection_mode" in cached_controls:
                normalized_mode = self._normalize_selection_mode(str(cached_controls["selection_mode"]))
                if normalized_mode is not None:
                    selection_mode_override = normalized_mode
                    selection_mode_source = "cache"
            if apply_threshold_globally_source == "default" and "apply_threshold_globally" in cached_controls:
                normalized_bool = self._normalize_boolean_value(str(cached_controls["apply_threshold_globally"]))
                if normalized_bool is not None:
                    apply_threshold_globally_override = normalized_bool
                    apply_threshold_globally_source = "cache"
            if max_returned_masks_source == "default" and "max_returned_masks" in cached_controls:
                try:
                    max_returned_masks = max(1, int(float(cached_controls["max_returned_masks"])))
                    max_returned_masks_source = "cache"
                except (TypeError, ValueError):
                    pass

        has_geo  = bool(point_coords) or bool(input_boxes)
        text_prompt_source = "none"
        source_priority = {"default": 0, "annotation": 1, "cache": 2, "context": 3}
        # With geometric cues, prefer mixed text prompt. If mixed is absent, pure text
        # still participates so geo+pure is a real, effective inference path.
        if has_geo:
            mixed_priority = source_priority.get(mixed_text_prompt_source, 0) if mixed_text_prompt else -1
            pure_priority = source_priority.get(pure_text_prompt_source, 0) if pure_text_prompt else -1
            if mixed_text_prompt and mixed_priority > pure_priority:
                text_prompt = mixed_text_prompt
                text_prompt_source = "mixed"
            elif pure_text_prompt and pure_priority > mixed_priority:
                text_prompt = pure_text_prompt
                text_prompt_source = "pure_geo"
            elif mixed_text_prompt:
                # Same source priority prefers mixed in geo flows.
                text_prompt = mixed_text_prompt
                text_prompt_source = "mixed"
            elif pure_text_prompt:
                text_prompt = pure_text_prompt
                text_prompt_source = "pure_geo"
            else:
                text_prompt = None
        else:
            if pure_text_prompt:
                text_prompt = pure_text_prompt
                text_prompt_source = "pure"
            else:
                text_prompt = None

        default_selection_mode = "all" if RETURN_ALL_MASKS else MASK_SELECTION_MODE
        selection_mode = (
            selection_mode_override
            or self._normalize_selection_mode(default_selection_mode)
            or "adaptive"
        )
        effective_apply_threshold_globally = (
            apply_threshold_globally_override
            if apply_threshold_globally_override is not None
            else APPLY_THRESHOLD_GLOBALLY
        )

        if runtime_cache_key:
            cache_payload: Dict[str, Any] = {}
            if pure_text_prompt and pure_text_prompt_source != "default":
                cache_payload["pure_text_prompt"] = pure_text_prompt
            if mixed_text_prompt and mixed_text_prompt_source != "default":
                cache_payload["mixed_text_prompt"] = mixed_text_prompt
            if confidence_threshold_source != "default":
                cache_payload["confidence_threshold"] = confidence_threshold
            if selection_mode_source != "default":
                cache_payload["selection_mode"] = selection_mode
            if apply_threshold_globally_source != "default":
                cache_payload["apply_threshold_globally"] = str(effective_apply_threshold_globally).lower()
            if max_returned_masks_source != "default":
                cache_payload["max_returned_masks"] = max_returned_masks
            if cache_payload:
                self._set_cached_runtime_controls(runtime_cache_key, cache_payload)

        logger.info(
            "Runtime controls resolved: threshold=%.4f(%s) selection_mode=%s(%s) apply_threshold_globally=%s(%s) max_returned_masks=%d(%s) text_prompt_source[pure=%s,mixed=%s]",
            confidence_threshold,
            confidence_threshold_source,
            selection_mode,
            selection_mode_source,
            effective_apply_threshold_globally,
            apply_threshold_globally_source,
            max_returned_masks,
            max_returned_masks_source,
            pure_text_prompt_source,
            mixed_text_prompt_source,
        )

        logger.debug(
            "text=%r  text_source=%s  points=%s  labels=%s  boxes=%s  label=%s  threshold=%.4f  selection_mode=%s  apply_threshold_globally=%s  max_returned_masks=%d",
            text_prompt,
            text_prompt_source,
            point_coords,
            point_labels,
            [(b, "+" if p else "-") for b, p in input_boxes], selected_label, confidence_threshold, selection_mode, effective_apply_threshold_globally, max_returned_masks,
        )

        has_text = ENABLE_PCS and text_prompt is not None

        if has_geo and ENABLE_PCS and text_prompt is None:
            logger.info(
                "Geo trigger without resolved text prompt: falling back to geo_only. "
                "Ensure mixed text is submitted (+ Add) or saved in annotation results."
            )

        if not has_text and not has_geo:
            return ModelResponse(predictions=[])

        # Text-only path: no geometric context → selected_label stays None.
        # Default to the first BrushLabels label so the mask gets a visible colour.
        if selected_label is None:
            brush_labels = self.parsed_label_config.get(from_name, {}).get("labels", [])
            if not brush_labels:
                for cfg in self.parsed_label_config.values():
                    cfg_labels = cfg.get("labels", [])
                    if cfg_labels:
                        brush_labels = cfg_labels
                        break
            selected_label = brush_labels[0] if brush_labels else "Object"

        # ── Load image ─────────────────────────────────────────────────────────
        _raw_url = tasks[0]["data"][value]
        logger.debug("Raw image URL from task data: %r", _raw_url)
        img_url  = _to_internal_url(_raw_url)
        logger.debug("After _to_internal_url: %r", img_url)
        _ls_base = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
        if _raw_url.startswith("s3://"):
            # Task data stores a bare s3:// URI (Cloud Storage import).
            # _to_internal_url intentionally leaves this unchanged now.
            # Convert to LS resolve URL so proxy mode serves the image.
            _fileuri  = base64.b64encode(_raw_url.encode()).decode()
            _task_id  = tasks[0].get("id")
            img_url   = f"{_ls_base}/tasks/{_task_id}/resolve/?fileuri={_fileuri}"
            logger.debug("S3 URL → resolve: %r", img_url)
            img_path = _download_ls_url(img_url)
        elif img_url.startswith("http://label-studio:") or img_url.startswith(_ls_base):
            # LS internal URL (resolve or local upload) — download with API auth
            img_path = _download_ls_url(img_url)
        else:
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
        assert image_width is not None and image_height is not None
        image_width = int(image_width)
        image_height = int(image_height)

        image = np.array(pil_img)

        # ── Run predictor ──────────────────────────────────────────────────────
        # If Agent is enabled, and there's a text prompt but no manual geometric hints, use the agent
        if AGENT_ENABLED and text_prompt and not point_coords and not input_boxes:
            try:
                return self._predict_sam3_agent(
                    img_path,
                    text_prompt,
                    selected_label,
                    image_width,
                    image_height,
                    from_name,
                    to_name,
                    confidence_threshold=confidence_threshold,
                    selection_mode=selection_mode,
                    apply_threshold_globally=effective_apply_threshold_globally,
                    max_returned_masks=max_returned_masks,
                )
            except Exception as exc:
                logger.error("SAM3 Agent prediction failed, falling back to basic text prompting: %s", exc, exc_info=True)

        # All paths go through Sam3Processor (text, geo-only, text+geo).
        try:
            return self._predict_sam3(
                    image, text_prompt, point_coords, point_labels, input_boxes,
                    selected_label, image_width, image_height,
                    from_name, to_name,
                    confidence_threshold=confidence_threshold,
                    selection_mode=selection_mode,
                    text_prompt_source=text_prompt_source,
                    apply_threshold_globally=effective_apply_threshold_globally,
                    max_returned_masks=max_returned_masks,
                )
        except Exception as exc:
            logger.error("Predict failed: %s", exc, exc_info=True)
            return ModelResponse(predictions=[])

    def fit(self, event: str, data: dict, **kwargs) -> None:
        logger.info("Received event '%s' (fit not implemented)", event)

    @staticmethod
    def _select_mask_indices(
        scores_tensor: Optional[torch.Tensor],
        n_total: int,
        *,
        has_text: bool,
        has_geo: bool,
        selection_mode: str,
        min_return_score: float,
        max_returned_masks: int,
    ) -> list[int]:
        """Choose which mask candidates should be returned to Label Studio."""
        if n_total <= 0:
            return []

        sorted_indices = list(range(n_total))
        if scores_tensor is not None:
            sorted_indices = sorted(
                range(n_total),
                key=lambda idx: float(scores_tensor[idx]),
                reverse=True,
            )

        if selection_mode == "all":
            selected = sorted_indices
        elif selection_mode == "top1":
            selected = sorted_indices[:1]
        elif selection_mode == "topk":
            selected = sorted_indices[: min(max_returned_masks, n_total)]
        elif selection_mode == "threshold":
            if scores_tensor is None:
                selected = sorted_indices[:1]
            else:
                selected = [
                    idx for idx in sorted_indices
                    if float(scores_tensor[idx]) >= min_return_score
                ]
        else:
            # adaptive mode
            if has_text and not has_geo:
                cap = min(n_total, max_returned_masks, 3)
                selected = sorted_indices[: max(1, cap)]
            elif has_geo and not has_text:
                cap = min(n_total, max_returned_masks, 2)
                selected = sorted_indices[: max(1, cap)]
            else:
                selected = sorted_indices[:1]

        if scores_tensor is not None and min_return_score > 0 and selection_mode != "threshold":
            selected = [idx for idx in selected if float(scores_tensor[idx]) >= min_return_score]

        if not selected and (selection_mode == "threshold" or min_return_score > 0):
            return []

        if not selected:
            selected = sorted_indices[:1]

        return selected

    # ── SAM3 path ──────────────────────────────────────────────────────────────

    def _predict_sam3_agent(
        self,
        img_path: str,
        text_prompt: str,
        selected_label: Optional[str],
        image_width: int,
        image_height: int,
        from_name: str,
        to_name: str,
        *,
        confidence_threshold: float,
        selection_mode: str,
        apply_threshold_globally: bool,
        max_returned_masks: int,
    ) -> ModelResponse:
        """Run SAM3 Agentic logic based on Llama / vLLM reasoning."""
        import tempfile
        import json
        from sam3.agent.agent_core import agent_inference
        from sam3.agent.client_llm import send_generate_request
        from sam3.model.box_ops import box_xyxy_to_xywh
        from sam3.train.masks_ops import robust_rle_encode
        from copy import deepcopy

        def agent_send_generate_request(messages):
            return send_generate_request(
                messages,
                server_url=AGENT_LLM_URL,
                model=AGENT_MODEL_NAME,
                api_key=AGENT_LLM_KEY,
                max_tokens=2048,
            )

        def agent_call_sam_service(image_path: str, text_prompt: str, output_folder_path: str = "sam_agent_cache"):
            """Local SAM service adapter for the agent that bypasses network requests."""
            logger.info("Agent requesting SAM3 for text prompt: %s", text_prompt)
            pil_img = Image.open(image_path).convert("RGB")
            orig_img_w, orig_img_h = pil_img.size

            # Run inference natively
            inference_state = self._processor.set_image(pil_img)
            inference_state = self._processor.set_text_prompt(state=inference_state, prompt=text_prompt)

            pred_boxes_xyxy = torch.stack([
                inference_state["boxes"][:, 0] / orig_img_w,
                inference_state["boxes"][:, 1] / orig_img_h,
                inference_state["boxes"][:, 2] / orig_img_w,
                inference_state["boxes"][:, 3] / orig_img_h,
            ], dim=-1)
            pred_boxes_xywh = box_xyxy_to_xywh(pred_boxes_xyxy).tolist()
            
            # Agent expects COCO RLE with 'counts' property unraveled
            pred_masks_rle = robust_rle_encode(inference_state["masks"].squeeze(1))
            pred_masks_counts = [m["counts"] for m in pred_masks_rle]

            outputs = {
                "orig_img_h": orig_img_h,
                "orig_img_w": orig_img_w,
                "pred_boxes": pred_boxes_xywh,
                "pred_masks": pred_masks_counts,
                "pred_scores": inference_state["scores"].tolist(),
                "original_image_path": image_path,
            }

            # Pre-filter by valid lengths as client_sam3 does
            valid_masks, valid_boxes, valid_scores = [], [], []
            for i, rle_str in enumerate(outputs["pred_masks"]):
                if len(rle_str) > 4:
                    valid_masks.append(rle_str)
                    valid_boxes.append(outputs["pred_boxes"][i])
                    valid_scores.append(outputs["pred_scores"][i])
            outputs["pred_masks"] = valid_masks
            outputs["pred_boxes"] = valid_boxes
            outputs["pred_scores"] = valid_scores

            # Save to temporary JSON for agent_inference to read
            os.makedirs(output_folder_path, exist_ok=True)
            tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=output_folder_path, delete=False)
            
            # The agent expects output_image_path to be in the JSON
            # so it can load the image with the highlighted masks to reason.
            tf_img = tempfile.NamedTemporaryFile(mode="wb", suffix=".png", dir=output_folder_path, delete=False)
            outputs["output_image_path"] = tf_img.name
            tf_img.close()
            
            from sam3.agent.viz import visualize
            viz_image = visualize(outputs)
            viz_image.save(outputs["output_image_path"])

            json.dump(outputs, tf)
            tf.close()
            return tf.name

        ctx = torch.autocast(**_autocast_kwargs) if _autocast_kwargs else None
        if ctx:
            ctx.__enter__()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                agent_history, final_output_dict, _ = agent_inference(
                    img_path,
                    text_prompt,
                    send_generate_request=agent_send_generate_request,
                    call_sam_service=agent_call_sam_service,
                    output_dir=temp_dir,
                    debug=False,
                )

                if "pred_scores" not in final_output_dict or not final_output_dict["pred_scores"]:
                    return ModelResponse(predictions=[])

                # Convert outputs back to Label Studio responses
                import pycocotools.mask as maskUtils
                from label_studio_converter import brush

                pred_scores = np.array(final_output_dict["pred_scores"])
                pred_masks_encoded = final_output_dict["pred_masks"] # These are raw counts strings

                # Since we don't have the size stored directly in pred_masks array from agent,
                # we must reconstruct the COCO dictionary to decode it
                results = []
                # Find which indices to select
                n_total = len(pred_scores)
                # Apply same threshold logic
                selected_indices = []
                for i in range(n_total):
                    if pred_scores[i] >= confidence_threshold:
                        selected_indices.append(i)
                if not selected_indices and n_total > 0:
                    selected_indices = [np.argmax(pred_scores)] # fallback to best if available

                # We map selected indices through Agent, we don't use max_returned_masks necessarily
                # wait, let's use the identical mask filtering logic if requested
                selected_indices_to_use = self._select_mask_indices(
                    scores_tensor=pred_scores,
                    n_total=n_total,
                    has_text=True,
                    has_geo=False,
                    selection_mode=selection_mode,
                    min_return_score=confidence_threshold if apply_threshold_globally else MIN_RETURN_SCORE,
                    max_returned_masks=max_returned_masks,
                )

                for idx in selected_indices_to_use:
                    score = float(pred_scores[idx])
                    coco_rle = {
                        "size": [image_height, image_width],
                        "counts": pred_masks_encoded[idx],
                    }
                    if isinstance(coco_rle["counts"], str):
                        coco_rle["counts"] = coco_rle["counts"].encode("utf-8")
                    
                    binary_mask = maskUtils.decode(coco_rle)
                    # Label studio formatting
                    ls_rle = brush.mask2rle(binary_mask * 255)
                    results.append({
                        "from_name": from_name,
                        "to_name": to_name,
                        "original_width": image_width,
                        "original_height": image_height,
                        "image_rotation": 0,
                        "value": {
                            "format": "rle",
                            "rle": ls_rle,
                            "brushlabels": [selected_label] if selected_label else [],
                        },
                        "score": score,
                        "type": "brushlabels",
                        "readonly": False,
                    })

                logger.info("SAM3 Agent completed. Found %d masks (kept %d) using Llama reasoning.", n_total, len(results))
                return ModelResponse(predictions=[{
                    "result": results,
                    "model_version": self.model_version + "-agent",
                }])

            except Exception as e:
                logger.error("Error in agent inner predict: %s", str(e), exc_info=True)
                raise
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)

    # ── SAM3 native path ───────────────────────────────────────────────────────

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
        *,
        confidence_threshold: float,
        selection_mode: str,
        text_prompt_source: str,
        apply_threshold_globally: bool = APPLY_THRESHOLD_GLOBALLY,
        max_returned_masks: int = MAX_RETURNED_MASKS,
    ) -> ModelResponse:
        """Run SAM3 Sam3Processor pipeline."""
        ctx = torch.autocast(**_autocast_kwargs) if _autocast_kwargs else None
        if ctx:
            ctx.__enter__()
        try:
            return self._predict_sam3_inner(
                image, text_prompt, point_coords, point_labels, input_boxes,
                selected_label, image_width, image_height, from_name, to_name,
                confidence_threshold=confidence_threshold,
                selection_mode=selection_mode,
                text_prompt_source=text_prompt_source,
                apply_threshold_globally=apply_threshold_globally,
                max_returned_masks=max_returned_masks,
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
        *,
        confidence_threshold: float,
        selection_mode: str,
        text_prompt_source: str,
        apply_threshold_globally: bool = APPLY_THRESHOLD_GLOBALLY,
        max_returned_masks: int = MAX_RETURNED_MASKS,
    ) -> ModelResponse:
        def _add_point_prompt_or_fallback(
            state_dict: dict,
            px: float,
            py: float,
            is_positive: bool,
        ) -> dict:
            """Add point prompt using SAM3 point embeddings, fallback to box approximation.

            Preferred path:
              - ensure language features exist ("visual" if geo-only)
              - append a normalized point into state["geometric_prompt"]
              - run grounding forward pass

            Fallback path:
              - represent point as a tiny box and call add_geometric_prompt
            """
            cx = px / image_width
            cy = py / image_height

            try:
                # Geometric-only prompts require the synthetic "visual" text token.
                if "language_features" not in state_dict.get("backbone_out", {}):
                    if _processor is not None and hasattr(_processor, "confidence_threshold"):
                        with _processor_runtime_lock:
                            previous_threshold = getattr(_processor, "confidence_threshold")
                            try:
                                setattr(_processor, "confidence_threshold", confidence_threshold)
                                state_dict = _processor.set_text_prompt(prompt="visual", state=state_dict)  # type: ignore[attr-defined]
                            finally:
                                setattr(_processor, "confidence_threshold", previous_threshold)
                    else:
                        state_dict = _processor.set_text_prompt(prompt="visual", state=state_dict)  # type: ignore[attr-defined]

                geom_prompt = state_dict.get("geometric_prompt")
                can_append_points = (
                    geom_prompt is not None
                    and hasattr(geom_prompt, "append_points")
                    and hasattr(_processor, "_forward_grounding")
                )

                if can_append_points:
                    prompt_obj: Any = geom_prompt
                    points = torch.tensor(
                        [cx, cy],
                        device=DEVICE,
                        dtype=torch.float32,
                    ).view(1, 1, 2)
                    labels = torch.tensor(
                        [1 if is_positive else 0],
                        device=DEVICE,
                        dtype=torch.long,
                    ).view(1, 1)

                    # Prompt uses seq-first, batch-second: [N, B, 2] / [N, B]
                    prompt_obj.append_points(points=points, labels=labels)
                    return _processor._forward_grounding(state_dict)  # type: ignore[attr-defined]

            except Exception as point_err:
                logger.warning(
                    "Native point prompt failed, fallback to box approximation: %s",
                    point_err,
                )

            # Fallback path for older/incompatible sam3 builds.
            eps = max(1e-4, POINT_FALLBACK_HALF_SIZE)
            logger.debug(
                "Point fallback to tiny box: cx=%.4f cy=%.4f half=%.4f positive=%s",
                cx,
                cy,
                eps,
                is_positive,
            )
            return _processor.add_geometric_prompt(  # type: ignore[attr-defined]
                box=[cx, cy, eps * 2.0, eps * 2.0],
                label=is_positive,
                state=state_dict,
            )

        def _set_text_prompt_with_threshold(
            state_dict: dict,
            prompt: str,
        ) -> dict:
            if _processor is None:
                raise RuntimeError("SAM3 processor not initialized")

            # Preferred path: pass threshold per call if the installed sam3 API supports it.
            try:
                return _processor.set_text_prompt(  # type: ignore[attr-defined]
                    prompt=prompt,
                    state=state_dict,
                    confidence_threshold=confidence_threshold,
                )
            except TypeError:
                # Fallback for sam3 builds where threshold is a processor attribute.
                pass

            if hasattr(_processor, "confidence_threshold"):
                with _processor_runtime_lock:
                    previous_threshold = getattr(_processor, "confidence_threshold")
                    try:
                        setattr(_processor, "confidence_threshold", confidence_threshold)
                        return _processor.set_text_prompt(prompt=prompt, state=state_dict)  # type: ignore[attr-defined]
                    finally:
                        setattr(_processor, "confidence_threshold", previous_threshold)

            return _processor.set_text_prompt(prompt=prompt, state=state_dict)  # type: ignore[attr-defined]

        state = _processor.set_image(Image.fromarray(image))  # type: ignore[attr-defined]

        has_text = text_prompt is not None
        has_geo = bool(input_boxes or point_coords)
        effective_max_returned_masks = max(1, int(max_returned_masks))
        if has_text and has_geo:
            inference_mode = "mixed_text_geo" if text_prompt_source == "mixed" else "text_geo"
        elif has_text:
            inference_mode = "text_only"
        elif has_geo:
            inference_mode = "geo_only"
        else:
            inference_mode = "none"
        active_selection_mode = self._normalize_selection_mode(selection_mode) or "adaptive"
        threshold_scope = "global" if apply_threshold_globally else "threshold-only"
        if apply_threshold_globally or active_selection_mode == "threshold":
            effective_score_gate = max(MIN_RETURN_SCORE, confidence_threshold)
        else:
            effective_score_gate = MIN_RETURN_SCORE

        # Text prompt (PCS)
        if has_text:
            assert text_prompt is not None
            state = _set_text_prompt_with_threshold(state, text_prompt)

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

        # Points: prefer native SAM3 point embeddings through geometric_prompt.append_points.
        # If unavailable in a specific sam3 build, fallback to tiny box approximation.
        for (px, py), lbl in zip(point_coords, point_labels):
            state = _add_point_prompt_or_fallback(
                state_dict=state,
                px=px,
                py=py,
                is_positive=bool(lbl),
            )

        masks_tensor  = state.get("masks")   # [N, 1, H, W] bool
        scores_tensor = state.get("scores")  # [N] float
        boxes_tensor  = state.get("boxes")   # [N, 4] float pixel xyxy (may be None)

        if masks_tensor is None or masks_tensor.shape[0] == 0:
            logger.info(
                "SAM3 returned no detections (mode=%s, selection_mode=%s, threshold=%.2f).",
                inference_mode,
                active_selection_mode,
                confidence_threshold,
            )
            return ModelResponse(predictions=[])

        n_total = masks_tensor.shape[0]
        # Log ALL detected candidates before filtering
        if scores_tensor is not None:
            for i in range(n_total):
                s = float(scores_tensor[i])
                b = boxes_tensor[i].cpu().tolist() if boxes_tensor is not None else None
                logger.info("  [SAM3] candidate %d  score=%.4f  box=%s", i, s, b)

        indices = self._select_mask_indices(
            scores_tensor,
            n_total,
            has_text=has_text,
            has_geo=has_geo,
            selection_mode=active_selection_mode,
            min_return_score=effective_score_gate,
            max_returned_masks=effective_max_returned_masks,
        )
        if not indices:
            logger.info(
                "SAM3 returned %d candidates, but none satisfied selection mode '%s' with threshold=%.4f (mode=%s, text_source=%s)",
                n_total,
                active_selection_mode,
                effective_score_gate,
                inference_mode,
                text_prompt_source,
            )
            return ModelResponse(predictions=[])

        results = []
        score_lines: list[str] = [
            f"mode={inference_mode}, text_source={text_prompt_source}, selection_mode={active_selection_mode}, threshold={effective_score_gate:.4f}, threshold_scope={threshold_scope}, topk_k={effective_max_returned_masks}"
        ]
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
            score_lines.append(
                f"(+{n_total - len(indices)} filtered candidates; mode={active_selection_mode})"
            )

        best_score = float(scores_tensor[indices].max()) if scores_tensor is not None else 1.0
        logger.info(
            "[SAM3][mode=%s][text_source=%s][selection_mode=%s] returning %d mask(s), best=%.4f",
            inference_mode,
            text_prompt_source,
            active_selection_mode,
            len(indices),
            best_score,
        )
        logger.info("Inference scores (mode=%s, selection_mode=%s):\n%s", inference_mode, active_selection_mode, "\n".join(score_lines))

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

    @staticmethod
    def _get_annotation_results_for_context(task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return annotation results as fallback control source for smart-trigger calls.

        Some Label Studio smart-tool flows send only geometric entries in
        context.result, omitting TextArea/Choices controls. This helper picks the
        matching annotation (by context.annotation_id when available) and returns
        its result list so runtime controls can still be resolved.
        """

        def _normalize_id(value: Any) -> Optional[str]:
            if value is None:
                return None
            return str(value)

        def _extract_result(annotation: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
            result = annotation.get("result") if isinstance(annotation, dict) else None
            if not isinstance(result, list):
                return []
            return [item for item in result if isinstance(item, dict)]

        context_annotation_id: Optional[str] = None
        if isinstance(context, dict):
            context_annotation_id = _normalize_id(context.get("annotation_id"))
            if context_annotation_id is None:
                annotation_obj = context.get("annotation")
                if isinstance(annotation_obj, dict):
                    context_annotation_id = _normalize_id(annotation_obj.get("id"))

        def _pick_target_annotation(annotations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            target: Optional[Dict[str, Any]] = None
            if context_annotation_id is not None:
                for annotation in annotations:
                    if _normalize_id(annotation.get("id")) == context_annotation_id:
                        target = annotation
                        break

            if target is None:
                candidates = [
                    annotation
                    for annotation in annotations
                    if isinstance(annotation.get("result"), list)
                    and annotation.get("result")
                ]
                if not candidates:
                    return None

                def _sort_key(annotation: Dict[str, Any]) -> str:
                    updated_at = annotation.get("updated_at")
                    created_at = annotation.get("created_at")
                    return str(updated_at or created_at or "")

                target = sorted(candidates, key=_sort_key)[-1]

            return target

        annotations = task.get("annotations") if isinstance(task, dict) else None
        if isinstance(annotations, list) and annotations:
            local_annotations = [item for item in annotations if isinstance(item, dict)]
            local_target = _pick_target_annotation(local_annotations)
            local_result = _extract_result(local_target)
            if local_result:
                return local_result

        if context_annotation_id is not None:
            annotation_payload = _ls_api_get_json(f"/api/annotations/{context_annotation_id}")
            annotation_result = _extract_result(annotation_payload)
            if annotation_result:
                logger.info(
                    "Recovered prompt controls from Label Studio annotation API (annotation_id=%s)",
                    context_annotation_id,
                )
                return annotation_result

        task_id = _normalize_id(task.get("id")) if isinstance(task, dict) else None
        if task_id is not None:
            task_payload = _ls_api_get_json(f"/api/tasks/{task_id}")
            task_annotations = task_payload.get("annotations") if isinstance(task_payload, dict) else None
            if isinstance(task_annotations, list) and task_annotations:
                remote_annotations = [item for item in task_annotations if isinstance(item, dict)]
                remote_target = _pick_target_annotation(remote_annotations)
                remote_result = _extract_result(remote_target)
                if remote_result:
                    logger.info(
                        "Recovered prompt controls from Label Studio task API (task_id=%s)",
                        task_id,
                    )
                    return remote_result

        return []

    @staticmethod
    def _runtime_controls_cache_key(task: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Optional[str]:
        def _normalize_id(value: Any) -> Optional[str]:
            if value is None:
                return None
            return str(value)

        task_id = _normalize_id(task.get("id")) if isinstance(task, dict) else None
        if task_id is None:
            return None

        annotation_id: Optional[str] = None
        if isinstance(context, dict):
            annotation_id = _normalize_id(context.get("annotation_id"))
            if annotation_id is None:
                annotation_obj = context.get("annotation")
                if isinstance(annotation_obj, dict):
                    annotation_id = _normalize_id(annotation_obj.get("id"))

        return f"{task_id}:{annotation_id or 'none'}"

    @staticmethod
    def _get_cached_runtime_controls(cache_key: str) -> Dict[str, Any]:
        with _runtime_controls_cache_lock:
            cache_value = _runtime_controls_cache.get(cache_key)
            if not isinstance(cache_value, dict):
                return {}
            return dict(cache_value)

    @staticmethod
    def _set_cached_runtime_controls(cache_key: str, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        with _runtime_controls_cache_lock:
            existing = _runtime_controls_cache.get(cache_key)
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(payload)
            _runtime_controls_cache[cache_key] = merged
            if len(_runtime_controls_cache) > _runtime_controls_cache_max_entries:
                # FIFO eviction: keep most recent entries and trim the oldest key.
                oldest_key = next(iter(_runtime_controls_cache))
                _runtime_controls_cache.pop(oldest_key, None)

    @staticmethod
    def _normalize_selection_mode(raw_mode: str) -> Optional[str]:
        mode = str(raw_mode or "").strip().lower()
        mode = SELECTION_MODE_ALIASES.get(mode, mode)
        if mode in VALID_SELECTION_MODES:
            return mode
        return None

    @staticmethod
    def _normalize_boolean_value(raw_value: str) -> Optional[bool]:
        value = str(raw_value or "").strip().lower()
        if value in {"1", "true", "yes", "on", "enabled", "enable", "global", "apply_threshold_globally"}:
            return True
        if value in {"0", "false", "no", "off", "disabled", "disable", "threshold-only", "threshold_only"}:
            return False
        return None

