"""SAM3 video tracking backend for Label Studio.

Architecture
------------
Same lazy-loading pattern as image backend — checkpoint downloaded at module
scope, model loaded on first predict() via _ensure_loaded(), after gunicorn
fork.  Do NOT use --preload.

Predictor selection (auto-detected at startup via import availability)
----------------------------------------------------------------------
  1. sam3.1 branch (preferred) — build_sam3_multiplex_video_predictor
       Returns Sam3MultiplexVideoPredictor; session-based API with PCS text prompts.
       Set SAM3_VIDEO_CHECKPOINT_FILENAME=sam3.1_multiplex.pt.

       Session lifecycle:
         resp       = pred.handle_request({"type": "start_session", "resource_path": …})
         session_id = resp["session_id"]
         pred.handle_request({"type": "add_prompt", "session_id": session_id,
                               "frame_index": 0, "obj_id": 0,
                               "bounding_boxes": [[x0, y0, w, h]],  # normalised xywh
                               "bounding_box_labels": [1],
                               "text": "optional PCS prompt"})
         for frame_data in pred.handle_stream_request(
                 {"type": "propagate_in_video", "session_id": session_id,
                  "propagation_direction": "forward",
                  "max_frame_num_to_track": N}):
             frame_idx = frame_data["frame_index"]
             outputs   = frame_data.get("outputs", {})
         pred.handle_request({"type": "close_session", "session_id": session_id})

  2. sam3 main branch — build_sam3_video_predictor
       Returns Sam3VideoPredictorMultiGPU; also extends Sam3BasePredictor so
       handle_request/handle_stream_request and PCS text prompts are supported.
       Set SAM3_VIDEO_CHECKPOINT_FILENAME=sam3.pt.
       No YAML config needed — SAM3 uses only checkpoint_path.


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


from typing import List, Dict, Optional
from urllib.parse import urlparse, urlunparse, parse_qs

import cv2
import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

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


# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_VIDEO_MODEL_ID", os.getenv("SAM3_MODEL_ID", "facebook/sam3.1"))
CHECKPOINT_FILENAME: str = os.getenv("SAM3_VIDEO_CHECKPOINT_FILENAME", os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.1_multiplex.pt"))
MAX_FRAMES_TO_TRACK: int = int(os.getenv("MAX_FRAMES_TO_TRACK", "10"))
# Cap the long side of extracted frames to reduce VRAM usage.
# SAM3's ViT attention is O(spatial_tokens²) — halving resolution cuts VRAM ~4×.
# 0 = no resize (original resolution). Recommended: 1024 for 1080p+ videos.
MAX_FRAME_LONG_SIDE: int = int(os.getenv("MAX_FRAME_LONG_SIDE", "1024"))

ENABLE_PCS: bool = os.getenv("SAM3_ENABLE_PCS", "true").lower() == "true"
# Flash Attention 3 — only effective when sam3 package is installed
ENABLE_FA3: bool = os.getenv("SAM3_ENABLE_FA3", "false").lower() == "true"

# ── CUDA optimisations ─────────────────────────────────────────────────────────
# TF32 and autocast setup is deferred to _ensure_loaded() (after gunicorn fork).
# Do NOT call torch.cuda.get_device_properties() here — it initialises CUDA in
# the gunicorn master process, causing RuntimeError in every forked worker.
_autocast_kwargs: Optional[dict] = None  # set in _ensure_loaded after fork

# ── Checkpoint download ────────────────────────────────────────────────────────
def _setup_precision() -> Optional[dict]:
    """Detect GPU compute capability and configure autocast precision.

    Precision tiers (based on minimum compute capability across all visible GPUs):

      sm_80+ Ampere (e.g. RTX 3060 Ti sm_86):
        bfloat16 autocast — SAM3 checkpoint weights are natively bf16; autocast
        ensures input activations are cast to bf16 to match, avoiding dtype
        mismatch.  allow_tf32 flags additionally accelerate any remaining fp32
        ops via TF32 Tensor Core.

      sm_70–79 Volta/Turing (e.g. TITAN V sm_70):
        bfloat16 autocast — SAM3 weights are natively bf16; must match for
        correctness.  BF16 has NO dedicated hardware on Volta/Turing (unlike
        Ampere), so bf16 ops are software-emulated (slower than fp16), but
        a dtype mismatch crash is worse than a performance penalty.

      sm_61 and below Pascal (e.g. GTX 1080):
        bfloat16 autocast — same addmm_act constraint as above.  SAM3 was not
        designed for Pascal; bf16 ops are software-emulated and performance
        will be poor, but the dtype mismatch must be resolved for correctness.

    Multi-GPU note:
      torch.autocast is a global context — it cannot be configured per-device.
      min_major across all visible GPUs determines the tier, so homogeneous
      GPU setups (all same generation) get the best-fit precision automatically.
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
    elif min_major >= 7:  # Volta/Turing (sm_70–79): FP16 Tensor Core native; no TF32
        logger.info("Precision: bfloat16 autocast — Volta (min sm_%d0, %d GPU(s))", min_major, n)
        return {"device_type": "cuda", "dtype": torch.bfloat16}
    else:  # Pascal (sm_60/61) and below: attempt bfloat16; may fail if addmm_act kernel missing
        logger.warning(
            "GPU compute capability sm_%d0 detected (Pascal or lower) — "
            "inference may fail if _addmm_activation bfloat16 kernel is absent on this GPU.",
            min_major,
        )
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

    logger.info("Downloading SAM3 video checkpoint '%s/%s' …", MODEL_ID, CHECKPOINT_FILENAME)
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
# Same rationale as sam3-image: defer CUDA init to after gunicorn fork.
_predictor = None
# "sam3_multiplex" = sam3.1 branch (build_sam3_multiplex_video_predictor, multiplex objects)
# "sam3_main"      = sam3 main branch (build_sam3_video_predictor, single GPU)
# Both use handle_request/handle_stream_request and support PCS text prompts.
_PREDICTOR_MODE: str = "sam3_multiplex"
_init_lock = threading.Lock()

_last_used: float = 0.0
_IDLE_TIMEOUT: int = int(os.getenv("GPU_IDLE_TIMEOUT_SECS", "3600"))  # default: 1 hour


def _idle_watchdog() -> None:
    """Daemon: unloads video predictor and frees VRAM after GPU_IDLE_TIMEOUT_SECS seconds idle."""
    global _predictor, _last_used
    while True:
        _time_module.sleep(60)
        if _predictor is None:
            continue
        if _time_module.monotonic() - _last_used > _IDLE_TIMEOUT:
            with _init_lock:
                if _predictor is not None and _time_module.monotonic() - _last_used > _IDLE_TIMEOUT:
                    logger.info("GPU idle >%ds — unloading video predictor to free VRAM.", _IDLE_TIMEOUT)
                    _predictor = None
                    if DEVICE.startswith("cuda"):
                        torch.cuda.empty_cache()
                        _gc.collect()


_watchdog_thread = threading.Thread(target=_idle_watchdog, daemon=True, name="gpu-idle-watchdog")
_watchdog_thread.start()


def _ensure_loaded() -> None:
    """Load SAM3 video predictor on first call inside the worker process."""
    global _predictor, _PREDICTOR_MODE
    if _predictor is not None:
        return
    with _init_lock:
        if _predictor is not None:
            return

        # ── Reset CUDA state for forked workers ───────────────────────────
        import torch.cuda as _cuda
        _cuda._in_bad_fork = False
        _cuda._initialized = False

        # ── CUDA precision (safe here — after fork, CUDA not yet init) ────────────
        # Precision: all CUDA GPUs → bfloat16 autocast (addmm_act constraint); Ampere+ also enables TF32.
        global _autocast_kwargs
        _autocast_kwargs = _setup_precision()

        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor  # type: ignore[import]

            # ── FA3 suppression ──────────────────────────────────────────────
            # sam3/model/model_misc.py::get_sdpa_settings() runs at MODULE import
            # time and sets USE_FLASH_ATTN=True on Ampere+ GPUs.  At inference,
            # the attention blocks read this flag and unconditionally execute:
            #   from sam3.perflib.fa3 import flash_attn_func
            # which in turn does:
            #   from flash_attn_interface import flash_attn_func as fa3
            # If flash-attn-3 is not installed (the default), this raises
            # ImportError during propagate_in_video — even when use_fa3=False
            # is passed to the builder (the builder param does not fully override
            # the module-level flag).
            # Fix: patch USE_FLASH_ATTN=False on the already-imported module object
            # before any forward pass, then also clear instance-level flags after build.
            if not ENABLE_FA3:
                try:
                    import sam3.model.model_misc as _sam3_misc  # type: ignore[import]
                    _sam3_misc.USE_FLASH_ATTN = False
                    logger.info("Patched sam3.model.model_misc.USE_FLASH_ATTN=False (FA3 disabled).")
                except Exception:
                    pass

            logger.info("Loading SAM3 multiplex video predictor on %s …", DEVICE)
            _predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=_checkpoint_path,
                use_fa3=ENABLE_FA3,
                async_loading_frames=True,
            )
            logger.info("SAM3 multiplex video predictor loaded.")

            # Belt-and-suspenders: patch instance-level use_fa3 / use_flash_attn
            # that may have been cached from USE_FLASH_ATTN during __init__.
            if not ENABLE_FA3:
                for _m in _predictor.modules():
                    for _attr in ("use_fa3", "use_flash_attn"):
                        if hasattr(_m, _attr):
                            setattr(_m, _attr, False)

        except ImportError as err:
            # ── sam3 main branch fallback ────────────────────────────────────
            # build_sam3_multiplex_video_predictor is sam3.1-only.
            # If missing, try sam3 main branch (Sam3VideoPredictorMultiGPU).
            # It also extends Sam3BasePredictor — same handle_request/
            # handle_stream_request API and PCS text prompt support.
            logger.warning(
                "sam3.1 multiplex predictor unavailable (%s) — trying sam3 main branch …",
                err,
            )
            try:
                from sam3.model_builder import build_sam3_video_predictor  # type: ignore[import]

                if not ENABLE_FA3:
                    try:
                        import sam3.model.model_misc as _sam3_misc  # type: ignore[import]
                        _sam3_misc.USE_FLASH_ATTN = False
                    except Exception:
                        pass

                logger.info("Loading SAM3 (main) video predictor on %s …", DEVICE)
                _predictor = build_sam3_video_predictor(
                    checkpoint_path=_checkpoint_path,
                    device=DEVICE,
                )
                _PREDICTOR_MODE = "sam3_main"
                logger.info("SAM3 (main) video predictor loaded.")

            except ImportError as err2:
                raise RuntimeError(
                    "Neither sam3.1 nor sam3 main branch is available. "
                    "Install the sam3 package from https://github.com/facebookresearch/sam3 . "
                    f"Last error: {err2}"
                ) from err2





# ── Backend ────────────────────────────────────────────────────────────────────

class NewModel(LabelStudioMLBase):
    """SAM3 video tracking backend for Label Studio.

    Workflow (SAM3 path):
      1. User draws VideoRectangle prompt on a frame (+ optional TextArea prompt).
      2. predict() resolves the video via get_local_path().
      3. SAM3 session is opened — predictor handles cv2 frame loading internally.
      4. add_prompt() is called with text + box per prompted frame.
      5. propagate_in_video() streams per-frame mask predictions.
      6. Masks → bboxes → VideoRectangle sequence merged with context and returned.

    Text prompt (PCS):
      If a TextArea result is present in context, its text is passed as "text"
      to add_prompt(). Works alongside or instead of geometric prompts.
      Passed to handle_request add_prompt as "text" field.
    """

    def setup(self) -> None:
        self.set("model_version", f"sam3-video:{MODEL_ID.split('/')[-1]}")

    def _resolve_default_track_label(self, from_name: str) -> str:
        """Resolve a default tracking label from Label Studio config."""
        labels = self.parsed_label_config.get(from_name, {}).get("labels", [])
        if labels:
            return str(labels[0])

        for cfg in self.parsed_label_config.values():
            cfg_labels = cfg.get("labels", [])
            if cfg_labels:
                return str(cfg_labels[0])

        return "Object"

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:

        _ensure_loaded()  # CUDA init deferred to worker process (after gunicorn fork)
        global _last_used
        _last_used = _time_module.monotonic()

        from_name, to_name, value = self.get_first_tag_occurence("VideoRectangle", "Video")

        if not context or not context.get("result"):
            return ModelResponse(predictions=[])

        task    = tasks[0]
        task_id = task.get("id")

        # ── Video metadata from VideoRectangle result (if present) ────────────
        # Text-only or KeyPoint-only paths have no VideoRectangle in context;
        # fps/frames_count/duration are probed from the video file in that case.
        vr_results   = [r for r in context["result"] if r.get("type") == "videorectangle"]
        frames_count: Optional[int]   = None
        duration:     Optional[float] = None
        fps:          Optional[float] = None
        if vr_results:
            _fc = vr_results[0]["value"].get("framesCount", 0)
            _dur = vr_results[0]["value"].get("duration", 1.0)
            frames_count = _fc
            duration     = _dur
            fps          = _fc / _dur if _dur else 25.0

        # ── Parse prompts ──────────────────────────────────────────────────────
        default_track_label = self._resolve_default_track_label(from_name)
        geo_prompts = self._get_geo_prompts(context, default_track_label)
        text_prompt = self._get_text_prompt(context)

        if not geo_prompts and not text_prompt:
            return ModelResponse(predictions=[])


        # ── Resolve video path ─────────────────────────────────────────────────
        _raw_video_url = task["data"][value]
        video_url = _to_internal_url(_raw_video_url)
        try:
            _ls_base = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
            if _raw_video_url.startswith("s3://"):
                # Task data stores a bare s3:// URI (Cloud Storage import).
                # Convert to LS resolve URL so proxy mode serves the video.
                _fileuri  = base64.b64encode(_raw_video_url.encode()).decode()
                video_url = f"{_ls_base}/tasks/{task_id}/resolve/?fileuri={_fileuri}"
                logger.debug("S3 URL → resolve: %r", video_url)
                video_path = _download_ls_url(video_url)
            elif video_url.startswith("http://label-studio:") or video_url.startswith(_ls_base):
                video_path = _download_ls_url(video_url)
            else:
                video_path = self.get_local_path(video_url, task_id=task_id)
        except Exception as exc:
            logger.error("Failed to resolve video path: %s", exc)
            return ModelResponse(predictions=[])

        # SAM3 start_session validates the file extension.
        # Label Studio's download_and_cache strips extensions (e.g. "abc123__").
        # If the cached file has no extension, create a symlink with the correct one.
        if not os.path.splitext(video_path)[1]:
            _parsed = urlparse(video_url)
            # Try query param ?d=path/to/file.mp4 first, then the URL path itself
            _d_param = parse_qs(_parsed.query).get("d", [""])[0]
            _ref_path = _d_param or _parsed.path
            _ext = os.path.splitext(_ref_path)[1] or ".mp4"
            _linked = video_path + _ext
            if not os.path.exists(_linked):
                os.symlink(video_path, _linked)
            logger.debug("Video path symlinked: %s → %s", video_path, _linked)
            video_path = _linked

        # Probe video for metadata when not available from VideoRectangle context
        # (text-only / KeyPoint-only paths).
        if fps is None:
            cap  = cv2.VideoCapture(video_path)
            _fps = cap.get(cv2.CAP_PROP_FPS)
            _n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            fps          = _fps if _fps > 0 else 25.0
            frames_count = _n
            duration     = _n / fps

        # ── Dispatch to correct predictor path ─────────────────────────────────
        score_lines: list[str] = []
        try:
            sequence, all_obj_ids, score_lines = self._predict_sam3(
                video_path, geo_prompts, text_prompt, fps,
            )
        except Exception as exc:
            logger.error("Video predict failed: %s", exc, exc_info=True)
            return ModelResponse(predictions=[])

        # ── Merge context sequence + new sequence ──────────────────────────────
        context_sequence = vr_results[0]["value"].get("sequence", []) if vr_results else []
        full_sequence    = context_sequence + sequence

        import uuid as _uuid
        result_items: list[dict] = [{
            "value": {
                "framesCount": frames_count,
                "duration":    duration,
                "sequence":    full_sequence,
            },
            "from_name": from_name,
            "to_name":   to_name,
            "type":      "videorectangle",
            "origin":    "manual",
            "id":        list(all_obj_ids)[0] if all_obj_ids else "obj0",
        }]

        # Scores TextArea — filled after each prediction (read-only display)
        if score_lines:
            logger.info("Inference scores:\n%s", "\n".join(score_lines))
        result_items.append({
            "id":        str(_uuid.uuid4())[:8],
            "from_name": "scores",
            "to_name":   to_name,
            "type":      "textarea",
            "value":     {"text": ["\n".join(score_lines) if score_lines else "—"]},
        })

        prediction = PredictionValue(result=result_items)
        return ModelResponse(predictions=[prediction])

    def fit(self, event: str, data: dict, **kwargs) -> None:
        logger.info("Received event '%s' (fit not implemented)", event)

    # ── SAM3 predict path ──────────────────────────────────────────────────────

    def _predict_sam3(
        self,
        video_path: str,
        geo_prompts: list[dict],
        text_prompt: Optional[str],
        fps: float,
    ) -> tuple[list[dict], set[str], list[str]]:
        """Run SAM3 session-based video predictor.

        Returns (sequence, all_obj_ids, score_lines).
        """
        import gc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        ctx = torch.autocast(**_autocast_kwargs) if _autocast_kwargs else None
        if ctx:
            ctx.__enter__()
        try:
            return self._predict_sam3_inner(video_path, geo_prompts, text_prompt, fps)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    def _predict_sam3_inner(
        self,
        video_path: str,
        geo_prompts: list[dict],
        text_prompt: Optional[str],
        fps: float,
    ) -> tuple[list[dict], set[str], list[str]]:
        all_obj_ids: set[str] = {p["obj_id"] for p in geo_prompts} or {"text_obj"}
        obj_id_map: dict[str, int] = {oid: i for i, oid in enumerate(all_obj_ids)}

        start_frame = min((p["frame_idx"] for p in geo_prompts), default=0)
        last_frame  = max((p["frame_idx"] for p in geo_prompts), default=0)
        score_lines: list[str] = []

        # ── Probe video dimensions ─────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if vid_w == 0 or vid_h == 0:
            logger.warning("Could not probe video dimensions for %s; boxes skipped.", video_path)
            vid_w = vid_h = 0

        # ── Extract only needed frames to avoid OOM on long videos ─────────
        # SAM3 start_session accepts an image folder (as well as a video file).
        # Extracting only [start_frame, last_frame + MAX_FRAMES_TO_TRACK] avoids
        # loading the entire video (e.g. 400+ frames × 1080p ≈ 2+ GB) into RAM.
        frame_end = last_frame + MAX_FRAMES_TO_TRACK + 1

        pred = _predictor
        assert pred is not None  # guaranteed by _ensure_loaded() above

        with tempfile.TemporaryDirectory() as frame_dir:
            n_extracted = self._extract_frames(video_path, frame_dir, start_frame, frame_end)
            if n_extracted == 0:
                logger.error(
                    "No frames extracted from %s [%d, %d)", video_path, start_frame, frame_end,
                )
                return [], all_obj_ids, score_lines

            logger.info(
                "Extracted %d frames [%d, %d) → %s (avoids loading full video)",
                n_extracted, start_frame, frame_end, frame_dir,
            )

            # Open session with image folder instead of full video file
            resp = pred.handle_request({
                "type":          "start_session",
                "resource_path": frame_dir,
            })
            session_id: str = resp["session_id"]

            try:
                # Text-only path: no geo_prompts → add_prompt at relative frame 0
                if not geo_prompts and text_prompt and ENABLE_PCS:
                    pred.handle_request({
                        "type":             "add_prompt",
                        "session_id":       session_id,
                        "frame_index":      0,
                        "obj_id":           0,
                        "text":             text_prompt,
                        "clear_old_boxes":  True,
                        "clear_old_points": True,
                    })

                # Group prompts by original frame_idx
                from collections import defaultdict
                prompts_by_frame: dict[int, list[dict]] = defaultdict(list)
                for p in geo_prompts:
                    prompts_by_frame[p["frame_idx"]].append(p)

                for orig_frame_idx in sorted(prompts_by_frame):
                    # Frame index relative to the extracted image folder
                    rel_frame_idx = orig_frame_idx - start_frame
                    frame_prompts = prompts_by_frame[orig_frame_idx]

                    # ── Build per-object prompt buckets ────────────────────
                    # box_entries: list of (xywh_normalized, is_positive)
                    # SAM3 multiplex predictor accepts ONLY bounding boxes — no
                    # point prompts. Keypoints are converted to tiny 2%×2% boxes.
                    by_obj: dict[str, dict] = defaultdict(lambda: {"box_entries": []})
                    box_obj_ids: list[str] = []

                    for p in frame_prompts:
                        if p["type"] == "box":
                            x0 = p["x_pct"] / 100.0
                            y0 = p["y_pct"] / 100.0
                            bw = p["w_pct"] / 100.0
                            bh = p["h_pct"] / 100.0
                            is_pos = p.get("is_positive", True)
                            by_obj[p["obj_id"]]["box_entries"].append(([x0, y0, bw, bh], is_pos))
                            if p["obj_id"] not in box_obj_ids:
                                box_obj_ids.append(p["obj_id"])
                        elif p["type"] == "point":
                            # Convert to tiny normalized box (2%×2%) centred on the point.
                            cx = p["x_pct"] / 100.0
                            cy = p["y_pct"] / 100.0
                            is_pos = bool(p.get("is_positive", True))
                            tiny = [cx, cy, 0.02, 0.02]
                            # Negative (background) points → attach to all box-defined
                            # objects; if none exist, attach to own obj_id.
                            if is_pos:
                                by_obj[p["obj_id"]]["box_entries"].append((tiny, True))
                            else:
                                targets = box_obj_ids if box_obj_ids else [p["obj_id"]]
                                for oid in targets:
                                    by_obj[oid]["box_entries"].append((tiny, False))

                    if not by_obj:
                        if text_prompt and ENABLE_PCS and frame_prompts:
                            pred.handle_request({
                                "type":             "add_prompt",
                                "session_id":       session_id,
                                "frame_index":      rel_frame_idx,
                                "obj_id":           obj_id_map[frame_prompts[0]["obj_id"]],
                                "text":             text_prompt,
                                "clear_old_boxes":  True,
                                "clear_old_points": True,
                            })
                        continue

                    for obj_id, data in by_obj.items():
                        req: dict = {
                            "type":             "add_prompt",
                            "session_id":       session_id,
                            "frame_index":      rel_frame_idx,
                            "obj_id":           obj_id_map[obj_id],
                            "clear_old_boxes":  True,
                            "clear_old_points": True,
                        }
                        if text_prompt and ENABLE_PCS:
                            req["text"] = text_prompt
                        entries = data["box_entries"]
                        if entries:
                            req["bounding_boxes"]      = [b for b, _ in entries]
                            req["bounding_box_labels"] = [1 if p else 0 for _, p in entries]
                        add_resp = pred.handle_request(req)
                        if add_resp:
                            logger.info(
                                "add_prompt (frame=%d→rel=%d obj=%d is_pos=%s): %s",
                                orig_frame_idx, rel_frame_idx,
                                obj_id_map[obj_id], data["is_positive"], add_resp,
                            )
                            score_lines.append(
                                f"frame={orig_frame_idx} obj={obj_id_map[obj_id]} "
                                f"{'[Object]' if data['is_positive'] else '[Exclude]'}: {add_resp}"
                            )

                rel_last_frame = last_frame - start_frame
                sequence: list[dict] = []

                try:
                    for frame_data in pred.handle_stream_request({
                        "type":                   "propagate_in_video",
                        "session_id":             session_id,
                        "propagation_direction":  "forward",
                        "start_frame_index":      rel_last_frame,
                        "max_frame_num_to_track": MAX_FRAMES_TO_TRACK,
                    }):
                        if frame_data is None:
                            continue
                        rel_idx: int   = frame_data["frame_index"]
                        outputs: dict  = frame_data.get("outputs") or {}
                        binary_masks: np.ndarray = outputs.get("out_binary_masks", np.array([]))

                        abs_frame = rel_idx + start_frame
                        for mask in binary_masks:
                            bbox = self._mask_to_bbox_pct(mask)
                            if bbox:
                                sequence.append({
                                    "frame":    abs_frame + 1,  # LS is 1-indexed
                                    "x":        bbox["x"],
                                    "y":        bbox["y"],
                                    "width":    bbox["width"],
                                    "height":   bbox["height"],
                                    "enabled":  True,
                                    "rotation": 0,
                                    "time":     rel_idx / fps,
                                })
                except Exception as _prop_err:
                    logger.warning(
                        "propagate_in_video raised an error (no detections or SAM3 internal): %s",
                        _prop_err,
                    )
            finally:
                pred.handle_request({
                    "type":       "close_session",
                    "session_id": session_id,
                })
            # TemporaryDirectory cleaned up here (after close_session)

        return sequence, all_obj_ids, score_lines

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_geo_prompts(self, context: dict, default_label: Optional[str] = None) -> list[dict]:
        """Parse VideoRectangle and KeyPointLabels items into prompt dicts.

        Returns a list of dicts with ``type`` = "box" or "point".
        Box fields  : obj_id, frame_idx, x_pct, y_pct, w_pct, h_pct, is_positive
        Point fields: obj_id, frame_idx, x_pct, y_pct, is_positive, label_name
        """
        if not default_label:
            default_label = self._resolve_default_track_label("")

        prompts: list[dict] = []
        for item in context.get("result", []):
            item_type = item.get("type")

            if item_type == "videorectangle":
                obj_id = item["id"]
                # Label applied to the whole tracking object (not per-frame sequence).
                # "Exclude" → negative prompt (bounding_box_labels=0).
                label_name  = (item["value"].get("labels") or [default_label])[0]
                is_positive = label_name.lower() != "exclude"
                for seq in item["value"].get("sequence", []):
                    if not seq.get("enabled", True):
                        continue
                    frame_idx = max(seq.get("frame", 1) - 1, 0)  # LS 1-indexed → 0-indexed
                    prompts.append({
                        "type":        "box",
                        "obj_id":      obj_id,
                        "frame_idx":   frame_idx,
                        "x_pct":       seq.get("x",      0.0),
                        "y_pct":       seq.get("y",      0.0),
                        "w_pct":       seq.get("width",  0.0),
                        "h_pct":       seq.get("height", 0.0),
                        "is_positive": is_positive,
                    })

            elif item_type == "keypointlabels":
                obj_id     = item["id"]
                val        = item["value"]
                # LS sends frame as 1-indexed for video controls; default to frame 1
                frame_idx  = max(val.get("frame", 1) - 1, 0)
                label_name = (val.get("keypointlabels") or [default_label])[0]
                is_positive = int(
                    item.get("is_positive", 0 if label_name.lower() == "background" else 1)
                )
                prompts.append({
                    "type":        "point",
                    "obj_id":      obj_id,
                    "frame_idx":   frame_idx,
                    "x_pct":       val.get("x", 0.0),
                    "y_pct":       val.get("y", 0.0),
                    "is_positive": is_positive,
                    "label_name":  label_name,
                })

        return sorted(prompts, key=lambda p: p["frame_idx"])

    def _get_text_prompt(self, context: dict) -> Optional[str]:
        """Extract the first non-empty TextArea value from context.

        Skips the scores TextArea (from_name="scores") to avoid feeding
        backend-generated score output back in as a text prompt.
        """
        for item in context.get("result", []):
            if item.get("type") == "textarea" and item.get("from_name") != "scores":
                texts = item["value"].get("text", [])
                if texts:
                    candidate = str(texts[0]).strip()
                    if candidate:
                        return candidate
        return None

    @staticmethod
    @staticmethod
    def _mask_to_bbox_pct(mask: np.ndarray) -> Optional[dict]:
        """Convert binary mask to percentage bounding box. Returns None if empty."""
        mask = mask.squeeze()
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        h, w = mask.shape
        return {
            "x":      round(float(xs.min()) / w * 100, 2),
            "y":      round(float(ys.min()) / h * 100, 2),
            "width":  round(float(xs.max() - xs.min() + 1) / w * 100, 2),
            "height": round(float(ys.max() - ys.min() + 1) / h * 100, 2),
        }

    @staticmethod
    def _extract_frames(
        video_path: str,
        frame_dir: str,
        start_frame: int,
        end_frame: int,
    ) -> int:
        """Extract frames [start_frame, end_frame) to frame_dir as JPEG images.

        Files are named 00000.jpg, 00001.jpg, … (relative index from start_frame).
        If MAX_FRAME_LONG_SIDE > 0, frames are downscaled so the long side ≤
        MAX_FRAME_LONG_SIDE before saving — cuts SAM3 ViT VRAM usage significantly
        (halving resolution reduces attention map memory ~4×).
        Returns the number of frames actually written.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video for frame extraction: %s", video_path)
            return 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        count = 0
        for rel_idx in range(end_frame - start_frame):
            ok, frame = cap.read()
            if not ok:
                break
            if MAX_FRAME_LONG_SIDE > 0:
                h, w = frame.shape[:2]
                long_side = max(h, w)
                if long_side > MAX_FRAME_LONG_SIDE:
                    scale = MAX_FRAME_LONG_SIDE / long_side
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(frame_dir, f"{rel_idx:05d}.jpg"), frame)
            count += 1
        cap.release()
        return count
