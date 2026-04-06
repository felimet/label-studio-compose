"""SAM3 video tracking backend for Label Studio.

Architecture
------------
Model is loaded at **module scope** (singleton) — same reason as image backend.

SAM3 video API (facebookresearch/sam3)
---------------------------------------
  predictor = Sam3VideoPredictorMultiGPU(checkpoint_path, use_fa3, ...)

  Session lifecycle per request:
    resp = predictor.handle_request({"type": "start_session",
                                     "resource_path": video_path})
    session_id = resp["session_id"]

    predictor.handle_request({"type": "add_prompt",
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

    for frame_data in predictor.handle_stream_request(
            {"type": "propagate_in_video",
             "session_id": session_id,
             "propagation_direction": "forward",
             "max_frame_num_to_track": N}):
        frame_idx = frame_data["frame_index"]
        outputs   = frame_data["outputs"]
        # outputs["out_obj_ids"]      ndarray [N]
        # outputs["out_binary_masks"] ndarray [N, H, W] bool
        # outputs["out_boxes_xywh"]   ndarray [N, 4]    pixel

    predictor.handle_request({"type": "close_session",
                               "session_id": session_id})

Key difference from SAM2
--------------------------
  - No manual cv2 frame splitting; predictor handles it via video_loader_type="cv2"
  - bounding_boxes are pixel [x0, y0, w, h] (NOT xyxy, NOT normalized)
  - Supports text prompt via "text" key in add_prompt
  - Flash Attention 3 enabled via use_fa3=True (requires flash-attn-3 installed)

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
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1")
CHECKPOINT_FILENAME: str = os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.1_multiplex.pt")
MODEL_DIR: str = os.getenv("MODEL_DIR", "/data/models")
MAX_FRAMES_TO_TRACK: int = int(os.getenv("MAX_FRAMES_TO_TRACK", "10"))

ENABLE_PCS: bool = os.getenv("SAM3_ENABLE_PCS", "true").lower() == "true"
# Flash Attention 3 — only effective when sam3 package is installed
ENABLE_FA3: bool = os.getenv("SAM3_ENABLE_FA3", "false").lower() == "true"

# ── CUDA optimisations ─────────────────────────────────────────────────────────
if DEVICE == "cuda":
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

_autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16} if DEVICE == "cuda" else None

# ── Checkpoint download ────────────────────────────────────────────────────────
try:
    from huggingface_hub import hf_hub_download

    _hf_token: Optional[str] = os.getenv("HF_TOKEN") or None
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("Downloading SAM3 video checkpoint '%s/%s' …", MODEL_ID, CHECKPOINT_FILENAME)
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
    from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU  # type: ignore[import]

    if ENABLE_FA3:
        logger.info("Flash Attention 3 enabled (use_fa3=True, use_rope_real=True).")

    logger.info("Loading SAM3 video predictor on %s (FA3=%s) …", DEVICE, ENABLE_FA3)
    predictor = Sam3VideoPredictorMultiGPU(
        checkpoint_path=_checkpoint_path,
        use_fa3=ENABLE_FA3,
        use_rope_real=ENABLE_FA3,   # FA3 requires real-valued RoPE
        async_loading_frames=True,
        video_loader_type="cv2",    # internal cv2 frame loader — no manual splitting
    )
    logger.info("SAM3 video predictor loaded.")

except ImportError:
    _USING_SAM2_FALLBACK = True
    logger.warning(
        "sam3 package not found — falling back to SAM2 video predictor. "
        "Text prompts (PCS) will be IGNORED. "
        "Install facebookresearch/sam3 when available."
    )
    ROOT_DIR = os.getcwd()
    sys.path.insert(0, ROOT_DIR)
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import]

    predictor = build_sam2_video_predictor(
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

        from_name, to_name, value = self.get_first_tag_occurence("VideoRectangle", "Video")

        if not context or not context.get("result"):
            return ModelResponse(predictions=[])

        task    = tasks[0]
        task_id = task.get("id")

        # ── Video metadata from first VideoRectangle result ────────────────────
        vr_results = [r for r in context["result"] if r.get("type") == "videorectangle"]
        if not vr_results:
            return ModelResponse(predictions=[])
        frames_count = vr_results[0]["value"].get("framesCount", 0)
        duration     = vr_results[0]["value"].get("duration", 1.0)
        fps          = frames_count / duration if duration else 25.0

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
        video_url = task["data"][value]
        try:
            video_path = self.get_local_path(video_url, task_id=task_id)
        except Exception as exc:
            logger.error("Failed to resolve video path: %s", exc)
            return ModelResponse(predictions=[])

        # ── Dispatch to correct predictor path ─────────────────────────────────
        try:
            if _USING_SAM2_FALLBACK:
                sequence, all_obj_ids = self._predict_sam2(
                    video_path, geo_prompts, fps,
                )
            else:
                sequence, all_obj_ids = self._predict_sam3(
                    video_path, geo_prompts, text_prompt, fps,
                )
        except Exception as exc:
            logger.error("Video predict failed: %s", exc, exc_info=True)
            return ModelResponse(predictions=[])

        # ── Merge context sequence + new sequence ──────────────────────────────
        context_sequence = vr_results[0]["value"].get("sequence", [])
        full_sequence    = context_sequence + sequence

        prediction = PredictionValue(
            result=[{
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
        )
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
    ) -> tuple[list[dict], set[str]]:
        """Run SAM3 session-based video predictor."""
        ctx = torch.autocast(**_autocast_kwargs) if _autocast_kwargs else None
        if ctx:
            ctx.__enter__()
        try:
            return self._predict_sam3_inner(video_path, geo_prompts, text_prompt, fps)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)

    def _predict_sam3_inner(
        self,
        video_path: str,
        geo_prompts: list[dict],
        text_prompt: Optional[str],
        fps: float,
    ) -> tuple[list[dict], set[str]]:
        all_obj_ids: set[str] = {p["obj_id"] for p in geo_prompts}
        obj_id_map: dict[str, int] = {oid: i for i, oid in enumerate(all_obj_ids)}

        start_frame = min((p["frame_idx"] for p in geo_prompts), default=0)

        # Open session — SAM3 handles internal frame loading via cv2
        resp = predictor.handle_request({
            "type":          "start_session",
            "resource_path": video_path,
        })
        session_id: str = resp["session_id"]

        try:
            # Probe video dimensions to convert percentage → pixel coords for add_prompt
            cap = cv2.VideoCapture(video_path)
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vid_w == 0 or vid_h == 0:
                logger.warning("Could not probe video dimensions for %s; boxes skipped.", video_path)
                vid_w = vid_h = 0

            # Group prompts by frame_idx
            from collections import defaultdict
            prompts_by_frame: dict[int, list[dict]] = defaultdict(list)
            for p in geo_prompts:
                prompts_by_frame[p["frame_idx"]].append(p)

            for frame_idx in sorted(prompts_by_frame):
                frame_prompts = prompts_by_frame[frame_idx]

                # Collect boxes for this frame (pixel [x0, y0, w, h])
                boxes: list[list[float]] = []
                box_labels: list[int]   = []
                if vid_w > 0 and vid_h > 0:
                    for p in frame_prompts:
                        x0 = p["x_pct"] / 100.0 * vid_w
                        y0 = p["y_pct"] / 100.0 * vid_h
                        bw = p["w_pct"] / 100.0 * vid_w
                        bh = p["h_pct"] / 100.0 * vid_h
                        boxes.append([x0, y0, bw, bh])
                        box_labels.append(1)

                req: dict = {
                    "type":             "add_prompt",
                    "session_id":       session_id,
                    "frame_index":      frame_idx,
                    "obj_id":           obj_id_map[frame_prompts[0]["obj_id"]],
                    "clear_old_boxes":  False,
                    "clear_old_points": False,
                }
                if text_prompt and ENABLE_PCS:
                    req["text"] = text_prompt
                if boxes:
                    req["bounding_boxes"]       = boxes
                    req["bounding_box_labels"]  = box_labels

                predictor.handle_request(req)

            # Propagate forward from the last prompted frame
            last_frame = max(p["frame_idx"] for p in geo_prompts)
            sequence: list[dict] = []

            for frame_data in predictor.handle_stream_request({
                "type":                   "propagate_in_video",
                "session_id":             session_id,
                "propagation_direction":  "forward",
                "start_frame_index":      last_frame,
                "max_frame_num_to_track": MAX_FRAMES_TO_TRACK,
            }):
                frame_idx: int      = frame_data["frame_index"]
                outputs: dict       = frame_data["outputs"]
                binary_masks: np.ndarray = outputs.get("out_binary_masks", np.array([]))

                for mask in binary_masks:
                    bbox = self._mask_to_bbox_pct(mask)
                    if bbox:
                        sequence.append({
                            "frame":    frame_idx + 1,      # LS is 1-indexed
                            "x":        bbox["x"],
                            "y":        bbox["y"],
                            "width":    bbox["width"],
                            "height":   bbox["height"],
                            "enabled":  True,
                            "rotation": 0,
                            "time":     (frame_idx - start_frame) / fps,
                        })
        finally:
            predictor.handle_request({
                "type":       "close_session",
                "session_id": session_id,
            })

        return sequence, all_obj_ids

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
            inference_state = predictor.init_state(video_path=frame_dir)
            predictor.reset_state(inference_state)

            for prompt in geo_prompts:
                pts, lbs = self._rect_to_keypoints(
                    prompt["x_pct"], prompt["y_pct"],
                    prompt["w_pct"], prompt["h_pct"],
                    width, height,
                )
                predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=prompt["frame_idx"],
                    obj_id=obj_id_map[prompt["obj_id"]],
                    points=pts,
                    labels=lbs,
                )

            sequence: list[dict] = []
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
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
        """Parse VideoRectangle items into prompt dicts."""
        prompts: list[dict] = []
        for item in context.get("result", []):
            if item.get("type") != "videorectangle":
                continue
            obj_id = item["id"]
            for seq in item["value"].get("sequence", []):
                if not seq.get("enabled", True):
                    continue
                frame_idx = max(seq.get("frame", 1) - 1, 0)  # LS 1-indexed → 0-indexed
                x_pct = seq.get("x",      0.0)
                y_pct = seq.get("y",      0.0)
                w_pct = seq.get("width",  0.0)
                h_pct = seq.get("height", 0.0)
                prompts.append({
                    "obj_id":    obj_id,
                    "frame_idx": frame_idx,
                    "x_pct":     x_pct,
                    "y_pct":     y_pct,
                    "w_pct":     w_pct,
                    "h_pct":     h_pct,
                    # Percentages converted to pixel [x0, y0, w, h] in _predict_sam3_inner
                    # after probing video dimensions via cv2.VideoCapture.
                })
        return sorted(prompts, key=lambda p: p["frame_idx"])

    def _get_text_prompt(self, context: dict) -> Optional[str]:
        """Extract the first non-empty TextArea value from context."""
        for item in context.get("result", []):
            if item.get("type") == "textarea":
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
