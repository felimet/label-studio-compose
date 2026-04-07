"""SAM3 video tracking backend for Label Studio.

Architecture
------------
Same lazy-loading pattern as image backend — checkpoint downloaded at module
scope, model loaded on first predict() via _ensure_loaded(), after gunicorn
fork.  Do NOT use --preload.

SAM3.1 video API (facebookresearch/sam3 @ sam3.1 branch)
----------------------------------------------------------
  predictor = build_sam3_multiplex_video_predictor(checkpoint_path, use_fa3=False, ...)
  # Returns Sam3MultiplexVideoPredictor (extends Sam3BasePredictor)

  Session lifecycle per request:
    resp = pred.handle_request({"type": "start_session",
                                     "resource_path": video_path})
    session_id = resp["session_id"]

    pred.handle_request({"type": "add_prompt",
                               "session_id": session_id,
                               "frame_index": 0,
                               "text": "optional text prompt",
                               "bounding_boxes": [[x0, y0, w, h]],   # pixel xywh
                               "bounding_box_labels": [1],
                               "points": [[px, py]],                  # pixel xy
                               "point_labels": [1],
                               "obj_id": int,
                               "clear_old_boxes": False,
                               "clear_old_points": False})

    for frame_data in pred.handle_stream_request(
            {"type": "propagate_in_video",
             "session_id": session_id,
             "propagation_direction": "both",
             "max_frame_num_to_track": N}):
        frame_idx = frame_data["frame_index"]
        outputs   = frame_data["outputs"]

    pred.handle_request({"type": "close_session",
                               "session_id": session_id})

Checkpoint mapping
-------------------
  sam3.pt             → Sam3VideoPredictorMultiGPU   (old, main branch)
  sam3.1_multiplex.pt → build_sam3_multiplex_video_predictor (sam3.1 branch, used here)

SAM2 fallback
--------------
  When sam3 package is not installed, falls back to SAM2 video predictor with
  the classic init_state / add_new_points / propagate_in_video interface.
  Text prompts are ignored in that mode.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
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
    internal = urlparse(ls_internal)
    # Only rewrite Label Studio paths — don't touch S3/MinIO/external URLs
    if parsed.path.startswith(("/data/", "/api/", "/tasks/")):
        return urlunparse(parsed._replace(scheme=internal.scheme, netloc=internal.netloc))
    return url


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
_USING_SAM2_FALLBACK: bool = False
_init_lock = threading.Lock()


def _ensure_loaded() -> None:
    """Load SAM3 (or SAM2 fallback) video predictor on first call inside the worker."""
    global _predictor, _USING_SAM2_FALLBACK
    if _predictor is not None:
        return
    with _init_lock:
        if _predictor is not None:
            return

        # ── Reset CUDA state for forked workers ───────────────────────────
        import torch.cuda as _cuda
        _cuda._in_bad_fork = False
        _cuda._initialized = False

        # ── CUDA optimisations (safe here — after fork, CUDA not yet init) ─
        global _autocast_kwargs
        if DEVICE == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # TF32 only effective on Ampere (sm_80+); safe to set on sm_70 (TITAN V),
            # PyTorch honours the flag only when hardware supports it.
            # bfloat16 autocast: valid on Volta (sm_70) and above.
            _autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16}

        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor  # type: ignore[import]

            logger.info("Loading SAM3 multiplex video predictor on %s …", DEVICE)
            _predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=_checkpoint_path,
                use_fa3=ENABLE_FA3,
                async_loading_frames=True,
            )
            logger.info("SAM3 multiplex video predictor loaded.")

        except ImportError as err:
            _USING_SAM2_FALLBACK = True
            logger.warning(
                "sam3 package import failed (%s) — falling back to SAM2 video predictor. "
                "Text prompts (PCS) will be IGNORED.",
                err,
            )
            _sam3_src = "/sam3"
            if _sam3_src not in sys.path:
                sys.path.insert(0, _sam3_src)
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import]

            _predictor = build_sam2_video_predictor(
                os.getenv("MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"),
                _checkpoint_path,
                device=DEVICE,
            )
            logger.info("SAM2 video predictor loaded (SAM3 fallback).")




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
      Ignored with WARNING when falling back to SAM2.
    """

    def setup(self) -> None:
        self.set("model_version", f"sam3-video:{MODEL_ID.split('/')[-1]}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:

        _ensure_loaded()  # CUDA init deferred to worker process (after gunicorn fork)

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
        geo_prompts = self._get_geo_prompts(context)
        text_prompt = self._get_text_prompt(context)

        if not geo_prompts and not text_prompt:
            return ModelResponse(predictions=[])


        if text_prompt and _USING_SAM2_FALLBACK:
            logger.warning(
                "Text prompt '%s' ignored — SAM2 fallback does not support PCS.",
                text_prompt,
            )
            text_prompt = None
            if not geo_prompts:
                return ModelResponse(predictions=[])

        # ── Resolve video path ─────────────────────────────────────────────────
        video_url = _to_internal_url(task["data"][value])
        try:
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
            if _USING_SAM2_FALLBACK:
                sequence, all_obj_ids = self._predict_sam2(
                    video_path, geo_prompts, fps,
                )
            else:
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

    # ── SAM2 fallback predict path ─────────────────────────────────────────────

    def _predict_sam2(
        self,
        video_path: str,
        geo_prompts: list[dict],
        fps: float,
    ) -> tuple[list[dict], set[str]]:
        """Run SAM2 video predictor (manual cv2 frame splitting)."""
        ctx = torch.autocast(**_autocast_kwargs) if _autocast_kwargs else None
        if ctx:
            ctx.__enter__()
        try:
            return self._predict_sam2_inner(video_path, geo_prompts, fps)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)

    def _predict_sam2_inner(
        self,
        video_path: str,
        geo_prompts: list[dict],
        fps: float,
    ) -> tuple[list[dict], set[str]]:
        all_obj_ids: set[str] = {p["obj_id"] for p in geo_prompts}
        obj_id_map: dict[str, int] = {oid: i for i, oid in enumerate(all_obj_ids)}

        first_frame = min(p["frame_idx"] for p in geo_prompts)
        last_frame  = max(p["frame_idx"] for p in geo_prompts)

        with tempfile.TemporaryDirectory() as frame_dir:
            frames = list(self._split_frames(
                video_path, frame_dir,
                start_frame=first_frame,
                end_frame=last_frame + MAX_FRAMES_TO_TRACK + 1,
            ))
            if not frames:
                logger.error("No frames extracted from video: %s", video_path)
                return [], all_obj_ids

            _frame_path, first_img = frames[0]
            height, width, _ = first_img.shape

            # Per-request state — no shared cache (not thread-safe with THREADS > 1)
            inference_state = _predictor.init_state(video_path=frame_dir)
            _predictor.reset_state(inference_state)

            # ── Build per-(frame, obj_id) point buckets ────────────────────
            # Boxes → converted to 5-point representation via _rect_to_keypoints.
            # Positive keypoints → appended directly.
            # Negative (background) keypoints → routed to box-defined objects on
            # the same frame; if none exist, routed to own obj_id.
            from collections import defaultdict
            prompts_by_frame_sam2: dict[int, list[dict]] = defaultdict(list)
            for p in geo_prompts:
                prompts_by_frame_sam2[p["frame_idx"]].append(p)

            by_frame_obj: dict[tuple, dict] = defaultdict(lambda: {"pts": [], "lbs": []})

            for frame_idx, frame_prompts in sorted(prompts_by_frame_sam2.items()):
                box_obj_ids_in_frame = [
                    p["obj_id"] for p in frame_prompts if p["type"] == "box"
                ]
                for p in frame_prompts:
                    key = (frame_idx, p["obj_id"])
                    if p["type"] == "box":
                        kp_pts, kp_lbs = self._rect_to_keypoints(
                            p["x_pct"], p["y_pct"],
                            p["w_pct"], p["h_pct"],
                            width, height,
                        )
                        by_frame_obj[key]["pts"].extend(kp_pts.tolist())
                        by_frame_obj[key]["lbs"].extend(kp_lbs.tolist())
                    elif p["type"] == "point":
                        px = p["x_pct"] / 100.0 * width
                        py = p["y_pct"] / 100.0 * height
                        if p["is_positive"]:
                            by_frame_obj[key]["pts"].append([px, py])
                            by_frame_obj[key]["lbs"].append(1)
                        else:
                            targets = box_obj_ids_in_frame if box_obj_ids_in_frame else [p["obj_id"]]
                            for oid in targets:
                                by_frame_obj[(frame_idx, oid)]["pts"].append([px, py])
                                by_frame_obj[(frame_idx, oid)]["lbs"].append(0)

            for (frame_idx, obj_id), data in sorted(by_frame_obj.items()):
                if not data["pts"]:
                    continue
                _predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id_map[obj_id],
                    points=np.array(data["pts"], dtype=np.float32),
                    labels=np.array(data["lbs"], dtype=np.int32),
                )

            sequence: list[dict] = []
            for out_frame_idx, out_obj_ids, out_mask_logits in _predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=last_frame,
                max_frame_num_to_track=MAX_FRAMES_TO_TRACK,
            ):
                real_frame = out_frame_idx + first_frame
                for i in range(len(out_obj_ids)):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    bbox = self._mask_to_bbox_pct(mask)
                    if bbox:
                        sequence.append({
                            "frame":    real_frame + 1,
                            "x":        bbox["x"],
                            "y":        bbox["y"],
                            "width":    bbox["width"],
                            "height":   bbox["height"],
                            "enabled":  True,
                            "rotation": 0,
                            "time":     out_frame_idx / fps,
                        })

        return sequence, all_obj_ids

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_geo_prompts(self, context: dict) -> list[dict]:
        """Parse VideoRectangle and KeyPointLabels items into prompt dicts.

        Returns a list of dicts with ``type`` = "box" or "point".
        Box fields  : obj_id, frame_idx, x_pct, y_pct, w_pct, h_pct, is_positive
        Point fields: obj_id, frame_idx, x_pct, y_pct, is_positive, label_name
        """
        prompts: list[dict] = []
        for item in context.get("result", []):
            item_type = item.get("type")

            if item_type == "videorectangle":
                obj_id = item["id"]
                # Label applied to the whole tracking object (not per-frame sequence).
                # "Exclude" → negative prompt (bounding_box_labels=0).
                label_name  = (item["value"].get("labels") or ["Object"])[0]
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
                label_name = (val.get("keypointlabels") or ["Object"])[0]
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
    def _rect_to_keypoints(
        x_pct: float, y_pct: float, w_pct: float, h_pct: float,
        width: int, height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert percentage rect to 5 keypoints (center + 4 cardinal midpoints)."""
        x = x_pct / 100.0
        y = y_pct / 100.0
        w = w_pct / 100.0
        h = h_pct / 100.0
        kps = [
            [x + w / 2,       y + h / 2      ],  # center
            [x + w / 4,       y + h / 2      ],  # left-center
            [x + 3 * w / 4,   y + h / 2      ],  # right-center
            [x + w / 2,       y + h / 4      ],  # top-center
            [x + w / 2,       y + 3 * h / 4  ],  # bottom-center
        ]
        pts = np.array(kps, dtype=np.float32)
        pts[:, 0] *= width
        pts[:, 1] *= height
        return pts, np.ones(len(kps), dtype=np.int32)

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

    @staticmethod
    def _split_frames(
        video_path: str,
        frame_dir: str,
        start_frame: int = 0,
        end_frame: int = 100,
    ):
        """cv2 frame splitter — only used in SAM2 fallback path."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return
        frame_count = rel = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_count < start_frame:
                frame_count += 1
                continue
            if frame_count >= end_frame:
                break
            fname = os.path.join(frame_dir, f"{rel:05d}.jpg")
            if not os.path.exists(fname):
                cv2.imwrite(fname, frame)
            yield fname, frame
            frame_count += 1
            rel += 1
        cap.release()
