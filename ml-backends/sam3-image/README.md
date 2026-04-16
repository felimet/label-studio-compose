# SAM3 Image Backend for Label Studio

Interactive image segmentation using [SAM 3.1](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) with Label Studio.

## Features

- **Text prompt** (SAM3 open-vocabulary PCS): label name → full-image detection of all matching instances
- **Point prompt**: KeyPointLabels (positive / negative clicks)
- **Box prompt**: RectangleLabels — any non-`Exclude` label is treated as positive; `Exclude` is a negative exemplar (`label=False`)
- **Output**: BrushLabels with Label Studio RLE encoding (label resolved dynamically from context / labeling config)
- **Scores display**: inference candidate scores written to `TextArea name="scores"` after each prediction
- **Automatic GPU precision**: Detects GPU compute capability and uses optimal precision (TF32, bfloat16, or fp32)
- **Multi-GPU support**: Uses accelerate `device_map="auto"` for automatic GPU dispatch (if available)
- **GPU idle release**: Automatically unloads model after inactivity timeout (default 1 hour)

## Prerequisites

1. NVIDIA GPU with driver ≥ 535.x (CUDA 12.6)
2. HuggingFace account with access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
   - Accept the model license at https://huggingface.co/facebook/sam3.1
   - Generate a token at https://huggingface.co/settings/tokens

## Quick Start

```bash
# 1. Set environment variables
cp ../../.env.example ../../.env
# Edit .env: set LABEL_STUDIO_API_KEY, HF_TOKEN

# 2. Start with main Label Studio stack
docker compose -f ../../docker-compose.yml -f ../../docker-compose.ml.yml up -d --build sam3-image-backend

# 3. Register in Label Studio UI
#    Project → Settings → Machine Learning → Add Model
#    URL: http://sam3-image-backend:9090
```

## Labeling Config

Use `labeling_config.xml` as your project's labeling interface.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SAM3_IMAGE_MODEL_ID` | `facebook/sam3.1` | HuggingFace model ID (image backend). Fallback: `SAM3_MODEL_ID` |
| `SAM3_IMAGE_CHECKPOINT_FILENAME` | `sam3.pt` | Checkpoint filename (~3.45 GB). No SAM3.1 image variant exists. Fallback: `SAM3_CHECKPOINT_FILENAME` |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `HF_TOKEN` | — | HuggingFace access token (required for gated model) |
| `LABEL_STUDIO_URL` | `http://label-studio:8080` | Label Studio internal URL |
| `LABEL_STUDIO_API_KEY` | — | Label Studio API token |
| `SAM3_ENABLE_PCS` | `true` | Enable natural-language text prompts (PCS). Set `false` for geometry-only mode |
| `SAM3_CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection score for text-prompt results (0–1) |
| `SAM3_RETURN_ALL_MASKS` | `false` | Return all detected instances (`true`) or only the top-scored one (`false`) |
| `SAM3_POINT_FALLBACK_HALF_SIZE` | `0.005` | Half-size of fallback tiny-box point prompt in normalized coordinates. Used only when native SAM3 point embeddings are unavailable in the runtime |
| `GPU_IDLE_TIMEOUT_SECS` | `3600` | Seconds of inactivity before model is unloaded from VRAM (default: 1 hour). Set lower (e.g., `300`) to release GPU memory more aggressively |

## Predict Paths

Three paths, all routed through `Sam3Processor` (no SAM2 fallback):

| Input | Path | Notes |
|-------|------|-------|
| TextArea only | Text-only PCS | `set_text_prompt()` → full-image detection, up to N masks. Image dimensions read from the loaded image (no geometric context required). |
| TextArea + geometry | Mixed | `set_text_prompt()` + box prompts via `add_geometric_prompt()` + keypoint prompts via native point embeddings |
| Geometry only | Geometric | box prompts via `add_geometric_prompt()`; keypoint prompts via native point embeddings |

> **Point prompts**: backend now prefers native SAM3 point embeddings (`geometric_prompt.append_points`) for KeyPoint prompts. Only when the current sam3 runtime lacks this path, it falls back to tiny-box approximation controlled by `SAM3_POINT_FALLBACK_HALF_SIZE`.

## References

- **Official SAM3 Repository**: https://github.com/facebookresearch/sam3
- **SAM3.1 Model Card (HuggingFace)**: https://huggingface.co/facebook/sam3.1
- **Label Studio ML Backend Examples**: https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image

For detailed SAM3/SAM3.1 architecture, checkpoints, and advanced configuration, refer to the official facebookresearch/sam3 repository and HuggingFace model card.

**GPU Hardware Notes**: Precision configuration adapts to your GPU automatically based on compute capability (Ampere sm_80+ uses TF32, Volta sm_70-79 uses bfloat16, Pascal sm_61 uses fp32).

## Running Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v --tb=short
```
