# SAM3 Image Backend for Label Studio

Interactive image segmentation using [SAM 3.1](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) with Label Studio.

## Features

- **Text prompt** (SAM3 open-vocabulary PCS): label name → find all matching instances
- **Point prompt**: KeyPointLabels (positive / negative clicks)
- **Box prompt**: RectangleLabels (bounding box)
- **Output**: BrushLabels with Label Studio RLE encoding

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
| `SAM3_MODEL_ID` | `facebook/sam3.1` | HuggingFace model ID |
| `SAM3_CHECKPOINT_FILENAME` | `sam3.1_multiplex.pt` | Checkpoint filename on HF |
| `MODEL_DIR` | `/data/models` | Local checkpoint cache directory |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `HF_TOKEN` | — | HuggingFace access token (required for gated model) |
| `LABEL_STUDIO_URL` | `http://label-studio:8080` | Label Studio internal URL |
| `LABEL_STUDIO_API_KEY` | — | Label Studio API token |
| `SAM3_ENABLE_PCS` | `true` | Enable natural-language text prompts (PCS). Set `false` for geometry-only mode |
| `SAM3_CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection score for text-prompt results (0–1) |
| `SAM3_RETURN_ALL_MASKS` | `false` | Return all detected instances (`true`) or only the top-scored one (`false`) |

## Predict Paths

Three paths, selectable by what is provided in the Label Studio context:

| Input | Path | Notes |
|-------|------|-------|
| TextArea only | Text-only PCS | `set_text_prompt()` → up to N masks |
| TextArea + geometry | Mixed | `set_text_prompt()` then `add_geometric_prompt()` |
| Geometry only | Geometric | `add_geometric_prompt()` per prompt |
| Any (SAM2 fallback) | SAM2 classic | Text ignored; geometric → `SAM2ImagePredictor.predict()` |

> **Point prompts**: `Sam3Processor` only accepts boxes, not points. Each KeyPoint is represented as a tiny 1%-sized box with `label=True/False` for positive/negative.

## Running Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v --tb=short
```
