"""SAM2.1 video tracking backend for Label Studio.

Architecture
------------
Checkpoints are downloaded at build time (download_models.py).
The model is loaded **lazily** on the first predict() call, after gunicorn
fork.  Do NOT use gunicorn --preload.

Model selection (persistent)
-----------------------------
Same mechanism as sam21-image: labeling_config.xml includes
<Choices name="sam_model">, resolved in priority order:

  1. context["result"] — Choices in current labeling session
  2. task["annotations"][-1]["result"] — last saved annotation
  3. /data/models/sam21_last_model.txt — persisted selection across restarts
  4. SAM21_DEFAULT_MODEL env var (default: sam2.1_hiera_large)

SAM2 video API (facebookresearch/sam2)
--------------------------------------
  predictor = build_sam2_video_predictor(config_file, ckpt_path, device=DEVICE)
  inference_state = predictor.init_state(video_path=frame_dir)
  predictor.add_new_points_or_box(
      inference_state,
      frame_idx=0,
      obj_id=0,
      box=np.array([x0, y0, x1, y1], dtype=np.float32),   # pixel coords
      points=np.array([[cx, cy]], dtype=np.float32),        # optional
      labels=np.array([1], dtype=np.int32),                 # 1=fg / 0=bg
  )
  for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
      # masks: [N_obj, 1, H, W] float32 logits (threshold at 0 for binary)
      ...
  predictor.reset_state(inference_state)
"""
from __future__ import annotations

import base64
import gc as _gc
import hashlib
import logging
import os
import tempfile
import threading
import time as _time_module
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

import cv2
import numpy as np
import requests
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/data/models"))
LAST_MODEL_FILE = MODEL_DIR / "sam21_last_model.txt"
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL = os.getenv("SAM21_DEFAULT_MODEL", "sam2.1_hiera_large")
GPU_IDLE_TIMEOUT = int(os.getenv("GPU_IDLE_TIMEOUT_SECS", "3600"))
MAX_FRAMES_TO_TRACK = int(os.getenv("MAX_FRAMES_TO_TRACK", "10"))
MAX_FRAME_LONG_SIDE = int(os.getenv("MAX_FRAME_LONG_SIDE", "1024"))

MODEL_CONFIGS: dict[str, str] = {
    "sam2.1_hiera_tiny":      "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large":     "configs/sam2.1/sam2.1_hiera_l.yaml",
}
CHECKPOINT_FILENAMES: dict[str, str] = {
    "sam2.1_hiera_tiny":      "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small":     "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus": "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large":     "sam2.1_hiera_large.pt",
}
VALID_MODELS = set(MODEL_CONFIGS.keys())


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


def _download_ls_url(url: str) -> str:
    """Download from a Label Studio internal URL with API token auth."""
    api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    headers = {"Authorization": f"Token {api_key}"} if api_key else {}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_dir = os.path.join(tempfile.gettempdir(), "ls-ml-video-cache")
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, f"{url_hash}.cache")
    if not os.path.exists(filepath):
        with open(filepath, "wb") as f:
            f.write(r.content)
    return filepath


def _load_predictor(model_key: str, device: str):
    """Load SAM2 video predictor for the given model key."""
    from sam2.build_sam import build_sam2_video_predictor

    ckpt_path = MODEL_DIR / CHECKPOINT_FILENAMES[model_key]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Run download_models.py or rebuild the Docker image."
        )

    config_file = MODEL_CONFIGS[model_key]
    logger.info(
        "Loading SAM2.1 video predictor: %s (config=%s, ckpt=%s)",
        model_key, config_file, ckpt_path,
    )
    predictor = build_sam2_video_predictor(config_file, str(ckpt_path), device=device)
    logger.info("SAM2.1 video predictor loaded: %s on %s", model_key, device)
    return predictor


def _detect_autocast_dtype(device: str) -> Optional[torch.dtype]:
    """Choose autocast dtype based on GPU compute capability."""
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
            return torch.bfloat16
        return torch.bfloat16  # Pascal: bf16 software emulated, needed for dtype consistency
    except Exception:
        return None


def _save_last_model(model_key: str) -> None:
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        LAST_MODEL_FILE.write_text(model_key)
    except Exception as exc:
        logger.warning("Could not save last model key: %s", exc)


def _read_last_model() -> Optional[str]:
    try:
        if LAST_MODEL_FILE.exists():
            key = LAST_MODEL_FILE.read_text().strip()
            if key in VALID_MODELS:
                return key
    except Exception:
        pass
    return None


# ── Main Model Class ───────────────────────────────────────────────────────────

class NewModel(LabelStudioMLBase):
    """SAM2.1 video tracking ML backend for Label Studio.

    Supports:
      - VideoRectangle prompts (Object = foreground, Exclude = background)
      - Persistent model selection via Choices checkbox in labeling_config.xml
      - Frame extraction to avoid OOM on long high-resolution videos
    """

    # Class-level state shared across requests in the same worker process
    _active_model_key: str = ""
    _predictor = None                    # SAM2VideoPredictor | None
    _autocast_dtype: Optional[torch.dtype] = None
    _lock: threading.Lock = threading.Lock()
    _last_activity: float = 0.0
    _idle_thread_started: bool = False

    def setup(self) -> None:
        self._start_idle_monitor()
        logger.info(
            "SAM2.1 video backend ready (default model: %s, device: %s)",
            DEFAULT_MODEL, DEVICE,
        )

    # ── Model lifecycle ───────────────────────────────────────────────────────

    def _ensure_model(self, model_key: str) -> None:
        """Load model if different from currently loaded one (thread-safe)."""
        with self._lock:
            if model_key == self._active_model_key and self._predictor is not None:
                logger.debug("Using cached video predictor: %s", model_key)
                return

            if self._predictor is not None:
                logger.info("Unloading video predictor: %s", self._active_model_key)
                del self._predictor
                self._predictor = None
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Reset CUDA state after gunicorn fork (safe on first call per worker)
            try:
                import torch.cuda as _cuda
                _cuda._in_bad_fork = False
                _cuda._initialized = False
            except Exception:
                pass

            self._predictor = _load_predictor(model_key, DEVICE)
            self._active_model_key = model_key
            self._autocast_dtype = _detect_autocast_dtype(DEVICE)
            _save_last_model(model_key)

    def _unload_model(self) -> None:
        with self._lock:
            if self._predictor is None:
                return
            logger.info(
                "GPU idle timeout (%ds): unloading video predictor %s",
                GPU_IDLE_TIMEOUT, self._active_model_key,
            )
            del self._predictor
            self._predictor = None
            self._active_model_key = ""
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _start_idle_monitor(self) -> None:
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

        t = threading.Thread(target=_monitor, daemon=True, name="sam21-video-idle-monitor")
        t.start()

    # ── Model selection ───────────────────────────────────────────────────────

    def _resolve_model_key(self, tasks: List[dict], context: Optional[dict]) -> str:
        if context:
            key = self._extract_choices_from_results(context.get("result", []))
            if key:
                return key
        for task in tasks:
            annotations = task.get("annotations") or []
            if annotations:
                key = self._extract_choices_from_results(annotations[-1].get("result", []))
                if key:
                    return key
        key = _read_last_model()
        if key:
            return key
        return DEFAULT_MODEL

    @staticmethod
    def _extract_choices_from_results(results: List[dict]) -> Optional[str]:
        for result in results:
            if result.get("from_name") == "sam_model" and result.get("type") == "choices":
                choices = result.get("value", {}).get("choices", [])
                if choices and choices[0] in VALID_MODELS:
                    return choices[0]
        return None

    # ── predict() ────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        NewModel._last_activity = _time_module.monotonic()

        if not context or not context.get("result"):
            return ModelResponse(predictions=[])

        from_name, to_name, value = self.get_first_tag_occurence("VideoRectangle", "Video")

        task = tasks[0]
        task_id = task.get("id")

        # VideoRectangle metadata (fps, frames_count)
        vr_results = [r for r in context["result"] if r.get("type") == "videorectangle"]
        frames_count: Optional[int] = None
        duration: Optional[float] = None
        fps: Optional[float] = None
        if vr_results:
            _fc = vr_results[0]["value"].get("framesCount", 0)
            _dur = vr_results[0]["value"].get("duration", 1.0)
            frames_count = _fc
            duration = _dur
            fps = _fc / _dur if _dur else 25.0

        # Parse prompts
        geo_prompts = self._get_geo_prompts(context)
        if not geo_prompts:
            return ModelResponse(predictions=[])

        # Resolve model and load
        model_key = self._resolve_model_key(tasks, context)
        self._ensure_model(model_key)

        # Resolve video path
        _raw_video_url = task["data"].get(value, "")
        video_url = _to_internal_url(_raw_video_url)
        try:
            _ls_base = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
            if _raw_video_url.startswith("s3://"):
                _fileuri = base64.b64encode(_raw_video_url.encode()).decode()
                video_url = f"{_ls_base}/tasks/{task_id}/resolve/?fileuri={_fileuri}"
                video_path = _download_ls_url(video_url)
            elif video_url.startswith("http://label-studio:") or video_url.startswith(_ls_base):
                video_path = _download_ls_url(video_url)
            else:
                video_path = self.get_local_path(video_url, task_id=task_id)
        except Exception as exc:
            logger.error("Failed to resolve video path: %s", exc)
            return ModelResponse(predictions=[])

        # Ensure video file has an extension (Label Studio cache may strip it)
        if not os.path.splitext(video_path)[1]:
            _parsed = urlparse(video_url)
            _d_param = parse_qs(_parsed.query).get("d", [""])[0]
            _ref_path = _d_param or _parsed.path
            _ext = os.path.splitext(_ref_path)[1] or ".mp4"
            _linked = video_path + _ext
            if not os.path.exists(_linked):
                os.symlink(video_path, _linked)
            video_path = _linked

        # Probe video metadata if not available from context
        if fps is None:
            cap = cv2.VideoCapture(video_path)
            _fps = cap.get(cv2.CAP_PROP_FPS)
            _n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            fps = _fps if _fps > 0 else 25.0
            frames_count = _n
            duration = _n / fps

        # Run inference
        try:
            sequence, all_obj_ids, info_lines = self._predict_sam2(
                video_path, geo_prompts, fps,
            )
        except Exception as exc:
            logger.error("Video predict failed: %s", exc, exc_info=True)
            return ModelResponse(predictions=[])

        # Merge context sequence with new predictions
        context_sequence = vr_results[0]["value"].get("sequence", []) if vr_results else []
        full_sequence = context_sequence + sequence

        import uuid as _uuid
        result_items: list[dict] = [{
            "value": {
                "framesCount": frames_count,
                "duration": duration,
                "sequence": full_sequence,
            },
            "from_name": from_name,
            "to_name": to_name,
            "type": "videorectangle",
            "origin": "manual",
            "id": list(all_obj_ids)[0] if all_obj_ids else "obj0",
        }]

        result_items.append({
            "id": str(_uuid.uuid4())[:8],
            "from_name": "scores",
            "to_name": to_name,
            "type": "textarea",
            "value": {"text": [f"model: {model_key}\n" + ("\n".join(info_lines) if info_lines else "—")]},
        })

        # Sync UI checkbox to the model that was actually used for this prediction.
        # Without this, Label Studio shows the XML default on tasks with no saved annotation.
        result_items.append({
            "id": str(_uuid.uuid4())[:8],
            "type": "choices",
            "value": {"choices": [model_key]},
            "from_name": "sam_model",
            "to_name": to_name,
        })

        prediction = PredictionValue(result=result_items)
        return ModelResponse(predictions=[prediction])

    def fit(self, event: str, data: dict, **kwargs) -> None:
        logger.info("Received event '%s' (fit not implemented)", event)

    # ── SAM2 predict path ─────────────────────────────────────────────────────

    def _predict_sam2(
        self,
        video_path: str,
        geo_prompts: list[dict],
        fps: float,
    ) -> tuple[list[dict], set[str], list[str]]:
        """Run SAM2 video predictor and return (sequence, obj_ids, info_lines)."""
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            _gc.collect()

        dtype = self._autocast_dtype
        if dtype is not None and DEVICE != "cpu":
            import contextlib
            ctx = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            import contextlib
            ctx = contextlib.nullcontext()

        with ctx:
            return self._predict_sam2_inner(video_path, geo_prompts, fps)

    def _predict_sam2_inner(
        self,
        video_path: str,
        geo_prompts: list[dict],
        fps: float,
    ) -> tuple[list[dict], set[str], list[str]]:
        predictor = self._predictor
        assert predictor is not None

        all_obj_ids: set[str] = {p["obj_id"] for p in geo_prompts}
        obj_id_map: dict[str, int] = {oid: i for i, oid in enumerate(sorted(all_obj_ids))}

        start_frame = min(p["frame_idx"] for p in geo_prompts)
        last_frame = max(p["frame_idx"] for p in geo_prompts)
        frame_end = last_frame + MAX_FRAMES_TO_TRACK + 1
        info_lines: list[str] = []

        # Probe video dimensions for coordinate scaling
        cap = cv2.VideoCapture(video_path)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        with tempfile.TemporaryDirectory() as frame_dir:
            n_extracted = self._extract_frames(video_path, frame_dir, start_frame, frame_end)
            if n_extracted == 0:
                logger.error("No frames extracted from %s [%d, %d)", video_path, start_frame, frame_end)
                return [], all_obj_ids, info_lines

            logger.info(
                "Extracted %d frames [%d, %d) → %s",
                n_extracted, start_frame, frame_end, frame_dir,
            )

            # SAM2 video predictor expects a directory of JPEG frames
            with torch.inference_mode():
                inference_state = predictor.init_state(video_path=frame_dir)

            try:
                # Add prompts per frame per object
                prompts_by_frame: dict[int, list[dict]] = defaultdict(list)
                for p in geo_prompts:
                    prompts_by_frame[p["frame_idx"]].append(p)

                for orig_frame_idx in sorted(prompts_by_frame):
                    rel_frame_idx = orig_frame_idx - start_frame
                    frame_prompts = prompts_by_frame[orig_frame_idx]

                    by_obj: dict[str, dict] = defaultdict(lambda: {
                        "boxes_fg": [], "boxes_bg": [], "pts_fg": [], "pts_bg": []
                    })

                    for p in frame_prompts:
                        oid = p["obj_id"]
                        if p["type"] == "box":
                            # Convert percentage coords to pixel coords
                            x0 = p["x_pct"] / 100.0 * vid_w
                            y0 = p["y_pct"] / 100.0 * vid_h
                            x1 = x0 + p["w_pct"] / 100.0 * vid_w
                            y1 = y0 + p["h_pct"] / 100.0 * vid_h
                            if p.get("is_positive", True):
                                by_obj[oid]["boxes_fg"].append([x0, y0, x1, y1])
                            else:
                                by_obj[oid]["boxes_bg"].append([x0, y0, x1, y1])
                        elif p["type"] == "point":
                            cx = p["x_pct"] / 100.0 * vid_w
                            cy = p["y_pct"] / 100.0 * vid_h
                            if p.get("is_positive", True):
                                by_obj[oid]["pts_fg"].append([cx, cy])
                            else:
                                by_obj[oid]["pts_bg"].append([cx, cy])

                    for obj_id, data in by_obj.items():
                        int_obj_id = obj_id_map[obj_id]

                        # Build box prompt (first FG box; BG boxes → bg points)
                        box_arr: Optional[np.ndarray] = None
                        if data["boxes_fg"]:
                            box_arr = np.array(data["boxes_fg"][0], dtype=np.float32)
                        # Convert BG boxes to background centre points
                        for bg_box in data["boxes_bg"]:
                            cx = (bg_box[0] + bg_box[2]) / 2
                            cy = (bg_box[1] + bg_box[3]) / 2
                            data["pts_bg"].append([cx, cy])

                        # Build point prompt
                        pts_all = data["pts_fg"] + data["pts_bg"]
                        labels_all = ([1] * len(data["pts_fg"])) + ([0] * len(data["pts_bg"]))
                        pts_arr: Optional[np.ndarray] = None
                        lbl_arr: Optional[np.ndarray] = None
                        if pts_all:
                            pts_arr = np.array(pts_all, dtype=np.float32)
                            lbl_arr = np.array(labels_all, dtype=np.int32)

                        if box_arr is None and pts_arr is None:
                            continue

                        with torch.inference_mode():
                            _, out_obj_ids, out_masks = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=rel_frame_idx,
                                obj_id=int_obj_id,
                                box=box_arr,
                                points=pts_arr,
                                labels=lbl_arr,
                            )
                        info_lines.append(
                            f"frame={orig_frame_idx} obj={int_obj_id} "
                            f"objs_out={out_obj_ids} "
                            f"masks_shape={[m.shape for m in out_masks]}"
                        )

                # Propagate forward
                sequence: list[dict] = []
                rel_last_frame = last_frame - start_frame

                with torch.inference_mode():
                    for rel_idx, obj_ids, masks in predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=rel_last_frame,
                        max_frame_num_to_track=MAX_FRAMES_TO_TRACK,
                    ):
                        abs_frame = rel_idx + start_frame
                        # masks: [N_obj, 1, H, W] float32 logits
                        binary_masks = (masks > 0).cpu().numpy()
                        for mask in binary_masks:
                            bbox = self._mask_to_bbox_pct(mask)
                            if bbox:
                                sequence.append({
                                    "frame": abs_frame + 1,  # LS is 1-indexed
                                    "x": bbox["x"],
                                    "y": bbox["y"],
                                    "width": bbox["width"],
                                    "height": bbox["height"],
                                    "enabled": True,
                                    "rotation": 0,
                                    "time": rel_idx / fps,
                                })

            finally:
                with torch.inference_mode():
                    predictor.reset_state(inference_state)

        logger.info(
            "SAM2.1 video (%s): %d sequence entries tracked",
            self._active_model_key, len(sequence),
        )
        return sequence, all_obj_ids, info_lines

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_geo_prompts(self, context: dict) -> list[dict]:
        """Parse VideoRectangle items into prompt dicts.

        Returns list of dicts:
          type='box':   obj_id, frame_idx, x_pct, y_pct, w_pct, h_pct, is_positive
          type='point': obj_id, frame_idx, x_pct, y_pct, is_positive
        """
        prompts: list[dict] = []
        for item in context.get("result", []):
            item_type = item.get("type")

            if item_type == "videorectangle":
                obj_id = item["id"]
                label_name = (item["value"].get("labels") or ["Object"])[0]
                is_positive = label_name.lower() != "exclude"
                for seq in item["value"].get("sequence", []):
                    if not seq.get("enabled", True):
                        continue
                    frame_idx = max(seq.get("frame", 1) - 1, 0)
                    prompts.append({
                        "type": "box",
                        "obj_id": obj_id,
                        "frame_idx": frame_idx,
                        "x_pct": seq.get("x", 0.0),
                        "y_pct": seq.get("y", 0.0),
                        "w_pct": seq.get("width", 0.0),
                        "h_pct": seq.get("height", 0.0),
                        "is_positive": is_positive,
                    })

            elif item_type == "keypointlabels":
                obj_id = item["id"]
                value = item.get("value", {})
                frame_idx = max(value.get("frame", 1) - 1, 0)
                labels = value.get("keypointlabels", [])
                is_positive = not (labels and labels[0] == "Exclude")
                prompts.append({
                    "type": "point",
                    "obj_id": obj_id,
                    "frame_idx": frame_idx,
                    "x_pct": value.get("x", 0.0),
                    "y_pct": value.get("y", 0.0),
                    "is_positive": is_positive,
                })

        return sorted(prompts, key=lambda p: p["frame_idx"])

    @staticmethod
    def _mask_to_bbox_pct(mask: np.ndarray) -> Optional[dict]:
        """Convert binary mask to percentage bounding box."""
        mask = mask.squeeze()
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        h, w = mask.shape
        return {
            "x": round(float(xs.min()) / w * 100, 2),
            "y": round(float(ys.min()) / h * 100, 2),
            "width": round(float(xs.max() - xs.min() + 1) / w * 100, 2),
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

        Files named 00000.jpg, 00001.jpg, … (relative index from start_frame).
        Downscales long side to MAX_FRAME_LONG_SIDE if > 0.
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
