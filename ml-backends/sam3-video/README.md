# SAM3 Video Backend for Label Studio

VideoRectangle tracking + optional text (PCS) prompts using [SAM 3.1](https://github.com/facebookresearch/sam3).

## Features

- **Text-only path**: TextArea prompt without a box → `add_prompt` at frame 0 with text only; SAM3 detects and tracks all matching instances forward.
- **Text + Box (mixed)**: text passed alongside VideoRectangle geometry for precise grounding.
- **Exclude box**: `Labels value="Exclude"` on a VideoRectangle → `bounding_box_labels=[0]` (negative prompt).
- **Scores display**: `add_prompt` responses written to `TextArea name="scores"` after each prediction.
- **SAM2 fallback**: if `sam3` package is unavailable, falls back to SAM2 video predictor; text prompts are ignored with a WARNING in that mode.

## Usage

1. Set `.env.ml`: `LABEL_STUDIO_API_KEY`, `HF_TOKEN`
2. `docker compose -f ../../docker-compose.yml -f ../../docker-compose.ml.yml up -d sam3-video-backend`
3. Label Studio: Add Model URL `http://sam3-video-backend:9090`
4. Use `labeling_config.xml` as project interface (includes `<TextArea>` for text prompts, `Exclude` label, scores area)

## Key Variables

| Variable | Default | Description |
|---|---|---|
| `SAM3_VIDEO_MODEL_ID` | `facebook/sam3.1` | HF model ID (video backend). Fallback: `SAM3_MODEL_ID` |
| `SAM3_VIDEO_CHECKPOINT_FILENAME` | `sam3.1_multiplex.pt` | Checkpoint filename (~3.5 GB). Requires `sam3.1` branch build arg. Fallback: `SAM3_CHECKPOINT_FILENAME` |
| `MAX_FRAMES_TO_TRACK` | `10` | Max frames to propagate per request |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `HF_TOKEN` | — | HF gated token (required) |
| `SAM3_ENABLE_PCS` | `true` | Enable text prompts. Supports text-only (no box required) and text+box mixed mode. Set `false` for geometry-only tracking. |
| `SAM3_ENABLE_FA3` | `false` | Flash Attention 3 (requires `--build-arg ENABLE_FA3=true` at build time) |

## Predict Paths

| Input | Path | Notes |
|-------|------|-------|
| TextArea only | Text-only PCS | `add_prompt` at frame 0 with `text` only; no box required. Video metadata probed from file via `cv2`. |
| VideoRectangle (Object) | Geometric | `bounding_box_labels=[1]` per prompted frame |
| VideoRectangle (Exclude) | Negative geometric | `bounding_box_labels=[0]`; tells SAM3 to exclude the region |
| TextArea + VideoRectangle | Mixed | `text` + `bounding_boxes` in same `add_prompt` call |

## Session Lifecycle

The SAM3 video predictor uses a session API:

```
start_session → add_prompt (per frame, with optional text + normalised xywh bbox) → propagate_in_video → close_session
```

`close_session` is guaranteed via `finally` even on errors. Video dimensions are probed via `cv2.VideoCapture` to convert percentage coordinates to normalised `[x0, y0, w, h]` required by `add_prompt`. For text-only requests (no VideoRectangle), metadata is also probed from the video file.

## Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v --tb=short
```
