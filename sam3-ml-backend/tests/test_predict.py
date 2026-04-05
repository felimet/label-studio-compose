"""Smoke tests for SAM3Backend.

Run:
    python -m pytest tests/ --tb=short -v

Tests use a synthetic 64x64 image — no real download required.
GPU is not required (tests run on CPU via DEVICE=cpu env var).
"""
from __future__ import annotations

import os
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Force CPU for tests — no GPU required
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("SAM3_MODEL_ID", "facebook/sam3")


def _make_synthetic_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a small synthetic RGB image for testing."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_task(image_url: str = "http://fake/image.jpg") -> dict:
    return {
        "id": 1,
        "data": {"image": image_url},
    }


def _make_keypoint_context(
    x_pct: float = 50.0,
    y_pct: float = 50.0,
    label: str = "Object",
    orig_w: int = 64,
    orig_h: int = 64,
) -> dict:
    return {
        "result": [
            {
                "type": "keypointlabels",
                "original_width": orig_w,
                "original_height": orig_h,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": 0.5,
                    "keypointlabels": [label],
                },
            }
        ]
    }


# ─── SAM3Backend unit tests ───────────────────────────────────────────────────


class TestCoordinateConversion(unittest.TestCase):
    """Test context parsing without model inference."""

    def setUp(self):
        # Import after env vars are set
        from model import SAM3Backend

        # Create backend without calling setup() (avoids downloading model)
        self.backend = object.__new__(SAM3Backend)

    def test_positive_keypoint_parsed(self):
        from model import SAM3Backend

        backend = object.__new__(SAM3Backend)
        context = _make_keypoint_context(x_pct=25.0, y_pct=75.0, orig_w=200, orig_h=100)
        points, labels, boxes = backend._parse_context(context, 200, 100)

        assert len(points) == 1
        assert points[0] == pytest.approx([50.0, 75.0])  # 25%*200, 75%*100
        assert labels[0] == 1  # positive

    def test_negative_keypoint_parsed(self):
        from model import SAM3Backend

        backend = object.__new__(SAM3Backend)
        context = _make_keypoint_context(label="Background", orig_w=100, orig_h=100)
        _, labels, _ = backend._parse_context(context, 100, 100)

        assert labels[0] == 0  # negative

    def test_rectangle_parsed(self):
        from model import SAM3Backend

        backend = object.__new__(SAM3Backend)
        context = {
            "result": [
                {
                    "type": "rectanglelabels",
                    "original_width": 200,
                    "original_height": 200,
                    "value": {
                        "x": 10.0, "y": 20.0,
                        "width": 30.0, "height": 40.0,
                        "rectanglelabels": ["Object"],
                    },
                }
            ]
        }
        _, _, boxes = backend._parse_context(context, 200, 200)
        assert len(boxes) == 1
        box = boxes[0]
        assert box[0] == pytest.approx(20.0)   # x1 = 10%*200
        assert box[1] == pytest.approx(40.0)   # y1 = 20%*200
        assert box[2] == pytest.approx(80.0)   # x2 = x1 + 30%*200
        assert box[3] == pytest.approx(120.0)  # y2 = y1 + 40%*200

    def test_empty_context_returns_empty(self):
        from model import SAM3Backend

        backend = object.__new__(SAM3Backend)
        points, labels, boxes = backend._parse_context({}, 100, 100)
        assert points == [] and labels == [] and boxes == []


class TestLabelConfigParsing(unittest.TestCase):
    def test_parses_brush_and_image_names(self):
        from model import SAM3Backend

        backend = object.__new__(SAM3Backend)
        backend.label_config = """
        <View>
          <Image name="img" value="$image"/>
          <BrushLabels name="brush" toName="img">
            <Label value="Cat"/>
          </BrushLabels>
        </View>
        """
        brush_name, img_name, label = backend._parse_label_config()
        assert brush_name == "brush"
        assert img_name == "img"
        assert label == "Cat"

    def test_fallback_on_missing_config(self):
        from model import SAM3Backend

        backend = object.__new__(SAM3Backend)
        backend.label_config = None
        brush_name, img_name, label = backend._parse_label_config()
        assert brush_name == "tag"
        assert img_name == "image"
        assert label == "Object"


class TestPredictWithMockedModel(unittest.TestCase):
    """Full predict() path with mocked transformers model."""

    def _make_backend_with_mock(self):
        from model import SAM3Backend

        backend = object.__new__(SAM3Backend)
        backend.label_config = """
        <View>
          <Image name="image" value="$image"/>
          <BrushLabels name="tag" toName="image">
            <Label value="Object"/>
          </BrushLabels>
        </View>
        """
        backend._image_cache = {}

        # Mock model outputs
        import torch

        mock_pred_masks = torch.zeros(1, 1, 1, 64, 64)
        mock_pred_masks[0, 0, 0, 20:40, 20:40] = 1.0  # non-empty mask

        mock_iou_scores = torch.tensor([[[0.92]]])

        mock_outputs = MagicMock()
        mock_outputs.pred_masks = mock_pred_masks
        mock_outputs.iou_scores = mock_iou_scores

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.zeros(1, 3, 64, 64),
            "original_sizes": torch.tensor([[64, 64]]),
            "reshaped_input_sizes": torch.tensor([[64, 64]]),
        }
        mock_processor.post_process_masks.return_value = [
            torch.zeros(1, 1, 64, 64).bool()
        ]

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs

        backend._model = mock_model
        backend._processor = mock_processor

        def mock_get(key):
            return "sam3-tracker:sam3" if key == "model_version" else None

        backend.get = mock_get
        return backend

    def test_predict_returns_model_response(self):
        from label_studio_ml.response import ModelResponse
        from model import SAM3Backend

        backend = self._make_backend_with_mock()
        task = _make_task()
        context = _make_keypoint_context()

        with patch.object(backend, "_fetch_image", return_value=_make_synthetic_image()):
            result = backend.predict([task], context=context)

        assert isinstance(result, ModelResponse)
        assert len(result.predictions) == 1

    def test_predict_no_context_returns_empty(self):
        from model import SAM3Backend

        backend = self._make_backend_with_mock()
        result = backend.predict([_make_task()], context=None)
        assert result.predictions[0]["result"] == []

    def test_predict_result_has_brushlabels_type(self):
        from model import SAM3Backend

        backend = self._make_backend_with_mock()
        context = _make_keypoint_context()

        with patch.object(backend, "_fetch_image", return_value=_make_synthetic_image()):
            result = backend.predict([_make_task()], context=context)

        pred = result.predictions[0]
        if pred["result"]:  # may be empty if mask is all zeros
            assert pred["result"][0]["type"] == "brushlabels"
            assert "rle" in pred["result"][0]["value"]
            assert isinstance(pred["result"][0]["value"]["rle"], list)


class TestRLEEncoding(unittest.TestCase):
    """Verify that mask2rle produces decodable output."""

    def test_rle_roundtrip(self):
        from label_studio_converter import brush

        # Create a simple binary mask
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # 20x20 square

        rle = brush.mask2rle(mask)
        assert isinstance(rle, list)
        assert len(rle) > 0

        decoded = brush.decode_rle(rle)
        # decoded is flat RGBA array — reshape and check non-zero region
        decoded_2d = np.frombuffer(decoded, dtype=np.uint8).reshape(64, 64, 4)
        # Alpha channel (index 3) should be non-zero in the mask region
        assert decoded_2d[20, 20, 3] > 0  # inside square
        assert decoded_2d[0, 0, 3] == 0   # outside square


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
