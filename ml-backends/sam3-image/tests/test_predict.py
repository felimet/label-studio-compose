"""Route-3 regression tests for SAM3 image backend."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("SAM3_MODEL_ID", "test/model")
os.environ.setdefault("SAM3_CHECKPOINT_FILENAME", "model.pt")
os.environ.setdefault("MODEL_DIR", "/tmp/test-models")
os.environ.setdefault("SAM3_ENABLE_PCS", "true")


_LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="brush" toName="image"><Label value="Object"/></BrushLabels>
  <KeyPointLabels name="kp" toName="image" smart="true"><Label value="Object"/></KeyPointLabels>
  <RectangleLabels name="rect" toName="image" smart="true"><Label value="Object"/></RectangleLabels>
    <TextArea name="text_prompt" toName="image" maxSubmissions="1" editable="true"/>
    <TextArea name="text_prompt_mixed" toName="image" maxSubmissions="1" editable="true"/>
    <TextArea name="confidence_threshold" toName="image" maxSubmissions="1" editable="true"/>
    <Choices name="apply_threshold_globally" toName="image" choice="multiple" showInline="true">
        <Choice value="apply_threshold_globally"/>
    </Choices>
    <Choices name="selection_mode" toName="image" choice="single" showInline="false">
        <Choice value="adaptive"/>
        <Choice value="top1"/>
        <Choice value="topk"/>
        <Choice value="threshold"/>
        <Choice value="all"/>
    </Choices>
    <TextArea name="selection_topk_k" toName="image" maxSubmissions="1" editable="true"/>
</View>
"""


def _backend():
    import model

    return model.NewModel(project_id="test", label_config=_LABEL_CONFIG)


def _make_task(url="/tmp/fake.jpg", task_id=1, annotations=None):
    task = {"id": task_id, "data": {"image": url}}
    if annotations is not None:
        task["annotations"] = annotations
    return task


def _text_context(text="the red car", from_name="text_prompt"):
    return {
        "result": [
            {
                "type": "textarea",
                "from_name": from_name,
                "value": {"text": [text]},
            }
        ]
    }


def _make_state(scores: list[float], h: int = 32, w: int = 32):
    masks = torch.zeros((len(scores), 1, h, w), dtype=torch.bool)
    boxes = []
    for i, _score in enumerate(scores):
        y0 = 2 + i
        x0 = 3 + i
        masks[i, 0, y0 : y0 + 6, x0 : x0 + 6] = True
        boxes.append([x0, y0, x0 + 6, y0 + 6])

    return {
        "masks": masks,
        "scores": torch.tensor(scores, dtype=torch.float32),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
    }


class TestCandidateSelection:
    def test_select_indices_adaptive_text_prefers_multiple_candidates(self):
        import model

        scores = torch.tensor([0.91, 0.81, 0.61, 0.21], dtype=torch.float32)
        indices = model.NewModel._select_mask_indices(
            scores,
            n_total=4,
            has_text=True,
            has_geo=False,
            selection_mode="adaptive",
            min_return_score=0.0,
            max_returned_masks=3,
        )

        assert indices == [0, 1, 2]

    def test_select_indices_topk_and_threshold(self):
        import model

        scores = torch.tensor([0.95, 0.75, 0.55], dtype=torch.float32)
        topk_indices = model.NewModel._select_mask_indices(
            scores,
            n_total=3,
            has_text=False,
            has_geo=True,
            selection_mode="topk",
            min_return_score=0.0,
            max_returned_masks=2,
        )

        threshold_indices = model.NewModel._select_mask_indices(
            scores,
            n_total=3,
            has_text=False,
            has_geo=True,
            selection_mode="threshold",
            min_return_score=0.7,
            max_returned_masks=3,
        )

        assert topk_indices == [0, 1]
        assert threshold_indices == [0, 1]

    def test_threshold_mode_can_return_empty_selection(self):
        import model

        scores = torch.tensor([0.45, 0.44, 0.43], dtype=torch.float32)
        indices = model.NewModel._select_mask_indices(
            scores,
            n_total=3,
            has_text=True,
            has_geo=False,
            selection_mode="threshold",
            min_return_score=0.8,
            max_returned_masks=3,
        )

        assert indices == []


class TestPredictInputTyping:
    def test_predict_converts_image_dimensions_to_int(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*args, **_kwargs):
            captured["image_width"] = args[6]
            captured["image_height"] = args[7]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=_text_context())

        assert isinstance(captured["image_width"], int)
        assert isinstance(captured["image_height"], int)
        assert captured["image_width"] == 60
        assert captured["image_height"] == 40

    def test_mixed_text_without_geo_does_not_trigger_inference(self):
        import model

        backend = _backend()
        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3") as predict_mock, \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            result = backend.predict(
                [_make_task()],
                context=_text_context(from_name="text_prompt_mixed"),
            )

        assert result.predictions == []
        predict_mock.assert_not_called()

    def test_predict_uses_pure_text_when_geo_exists_and_mixed_missing(self):
        import model

        backend = _backend()
        captured = {}
        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["bridge crack"]},
                },
                {
                    "type": "rectanglelabels",
                    "original_width": 60,
                    "original_height": 40,
                    "value": {
                        "x": 10.0,
                        "y": 10.0,
                        "width": 30.0,
                        "height": 20.0,
                        "rectanglelabels": ["Object"],
                    },
                },
            ]
        }

        def _fake_predict(*args, **kwargs):
            captured["text_prompt"] = args[1]
            captured["text_prompt_source"] = kwargs["text_prompt_source"]
            return model.ModelResponse(predictions=[])

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["text_prompt"] == "bridge crack"
        assert captured["text_prompt_source"] == "pure_geo"

    def test_predict_uses_context_confidence_threshold_when_provided(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["confidence_threshold"] = kwargs["confidence_threshold"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "textarea",
                    "from_name": "confidence_threshold",
                    "value": {"text": ["0.73"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["confidence_threshold"] == pytest.approx(0.73)

    def test_predict_uses_context_apply_threshold_globally_when_checked(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["apply_threshold_globally"] = kwargs["apply_threshold_globally"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "choices",
                    "from_name": "apply_threshold_globally",
                    "value": {"choices": ["apply_threshold_globally"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["apply_threshold_globally"] is True

    def test_predict_uses_context_apply_threshold_globally_false_when_cleared(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["apply_threshold_globally"] = kwargs["apply_threshold_globally"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "choices",
                    "from_name": "apply_threshold_globally",
                    "value": {"choices": []},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(model, "APPLY_THRESHOLD_GLOBALLY", True), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["apply_threshold_globally"] is False

    def test_predict_uses_context_selection_mode_when_provided(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["selection_mode"] = kwargs["selection_mode"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "choices",
                    "from_name": "selection_mode",
                    "value": {"choices": ["threshold"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["selection_mode"] == "threshold"

    def test_predict_uses_default_selection_mode_all_without_override(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["selection_mode"] = kwargs["selection_mode"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                }
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(model, "RETURN_ALL_MASKS", False), \
             patch.object(model, "MASK_SELECTION_MODE", "all"), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["selection_mode"] == "all"

    def test_predict_uses_context_topk_k_when_provided(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["max_returned_masks"] = kwargs["max_returned_masks"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "textarea",
                    "from_name": "selection_topk_k",
                    "value": {"text": ["5"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["max_returned_masks"] == 5

    def test_predict_clamps_topk_k_to_minimum_one(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["max_returned_masks"] = kwargs["max_returned_masks"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "textarea",
                    "from_name": "selection_topk_k",
                    "value": {"text": ["0"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["max_returned_masks"] == 1

    def test_predict_falls_back_to_default_topk_k_on_invalid_value(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["max_returned_masks"] = kwargs["max_returned_masks"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "textarea",
                    "from_name": "selection_topk_k",
                    "value": {"text": ["2.5"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(model, "MAX_RETURNED_MASKS", 4), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["max_returned_masks"] == 4

    def test_predict_supports_legacy_max_returned_masks_field_name(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["max_returned_masks"] = kwargs["max_returned_masks"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "textarea",
                    "from_name": "max_returned_masks",
                    "value": {"text": ["6"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["max_returned_masks"] == 6

    def test_predict_clamps_out_of_range_confidence_threshold(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["confidence_threshold"] = kwargs["confidence_threshold"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "textarea",
                    "from_name": "confidence_threshold",
                    "value": {"text": ["1.70"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["confidence_threshold"] == pytest.approx(1.0)

    def test_predict_falls_back_to_default_threshold_on_invalid_input(self):
        import model

        backend = _backend()
        captured = {}

        def _fake_predict(*_args, **kwargs):
            captured["confidence_threshold"] = kwargs["confidence_threshold"]
            return model.ModelResponse(predictions=[])

        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["car"]},
                },
                {
                    "type": "textarea",
                    "from_name": "confidence_threshold",
                    "value": {"text": ["invalid"]},
                },
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(model, "CONFIDENCE_THRESHOLD", 0.42), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["confidence_threshold"] == pytest.approx(0.42)

    def test_predict_keeps_legacy_custom_textarea_name_as_pure_text_prompt(self):
        import model

        backend = _backend()
        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "custom_prompt",
                    "value": {"text": ["bridge crack"]},
                }
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", return_value=model.ModelResponse(predictions=[])) as predict_mock, \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert predict_mock.call_args.args[1] == "bridge crack"

    def test_predict_marks_mixed_prompt_source_when_geo_exists(self):
        import model

        backend = _backend()
        captured = {}
        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt_mixed",
                    "value": {"text": ["bridge crack"]},
                },
                {
                    "type": "rectanglelabels",
                    "original_width": 60,
                    "original_height": 40,
                    "value": {
                        "x": 10.0,
                        "y": 10.0,
                        "width": 30.0,
                        "height": 20.0,
                        "rectanglelabels": ["Object"],
                    },
                },
            ]
        }

        def _fake_predict(*_args, **kwargs):
            captured["text_prompt_source"] = kwargs["text_prompt_source"]
            return model.ModelResponse(predictions=[])

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["text_prompt_source"] == "mixed"

    def test_predict_recovers_controls_from_annotation_when_context_omits_them(self):
        import model

        backend = _backend()
        captured = {}
        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "annotation_id": 99,
            "result": [
                {
                    "type": "rectanglelabels",
                    "original_width": 60,
                    "original_height": 40,
                    "value": {
                        "x": 10.0,
                        "y": 10.0,
                        "width": 30.0,
                        "height": 20.0,
                        "rectanglelabels": ["Object"],
                    },
                }
            ],
        }
        task_with_annotation_controls = _make_task(
            annotations=[
                {
                    "id": 99,
                    "result": [
                        {
                            "type": "textarea",
                            "from_name": "text_prompt_mixed",
                            "value": {"text": ["bridge crack"]},
                        },
                        {
                            "type": "textarea",
                            "from_name": "confidence_threshold",
                            "value": {"text": ["0.6"]},
                        },
                        {
                            "type": "choices",
                            "from_name": "selection_mode",
                            "value": {"choices": ["all"]},
                        },
                        {
                            "type": "textarea",
                            "from_name": "selection_topk_k",
                            "value": {"text": ["7"]},
                        },
                    ],
                }
            ]
        )

        def _fake_predict(*args, **kwargs):
            captured["text_prompt"] = args[1]
            captured["text_prompt_source"] = kwargs["text_prompt_source"]
            captured["confidence_threshold"] = kwargs["confidence_threshold"]
            captured["selection_mode"] = kwargs["selection_mode"]
            captured["max_returned_masks"] = kwargs["max_returned_masks"]
            return model.ModelResponse(predictions=[])

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([task_with_annotation_controls], context=context)

        assert captured["text_prompt"] == "bridge crack"
        assert captured["text_prompt_source"] == "mixed"
        assert captured["confidence_threshold"] == pytest.approx(0.6)
        assert captured["selection_mode"] == "all"
        assert captured["max_returned_masks"] == 7

    def test_predict_keeps_context_pure_text_over_annotation_mixed_text(self):
        import model

        backend = _backend()
        captured = {}
        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "annotation_id": 100,
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt",
                    "value": {"text": ["fresh pure prompt"]},
                },
                {
                    "type": "rectanglelabels",
                    "original_width": 60,
                    "original_height": 40,
                    "value": {
                        "x": 10.0,
                        "y": 10.0,
                        "width": 30.0,
                        "height": 20.0,
                        "rectanglelabels": ["Object"],
                    },
                },
            ],
        }
        task_with_annotation_controls = _make_task(
            annotations=[
                {
                    "id": 100,
                    "result": [
                        {
                            "type": "textarea",
                            "from_name": "text_prompt_mixed",
                            "value": {"text": ["stale mixed prompt"]},
                        },
                        {
                            "type": "textarea",
                            "from_name": "confidence_threshold",
                            "value": {"text": ["0.6"]},
                        },
                    ],
                }
            ]
        )

        def _fake_predict(*args, **kwargs):
            captured["text_prompt"] = args[1]
            captured["text_prompt_source"] = kwargs["text_prompt_source"]
            captured["confidence_threshold"] = kwargs["confidence_threshold"]
            return model.ModelResponse(predictions=[])

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([task_with_annotation_controls], context=context)

        assert captured["text_prompt"] == "fresh pure prompt"
        assert captured["text_prompt_source"] == "pure_geo"
        assert captured["confidence_threshold"] == pytest.approx(0.6)

    def test_predict_recovers_controls_from_annotation_api_when_task_annotations_missing(self):
        import model

        backend = _backend()
        captured = {}
        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        context = {
            "annotation_id": 321,
            "result": [
                {
                    "type": "rectanglelabels",
                    "original_width": 60,
                    "original_height": 40,
                    "value": {
                        "x": 10.0,
                        "y": 10.0,
                        "width": 30.0,
                        "height": 20.0,
                        "rectanglelabels": ["Object"],
                    },
                }
            ],
        }

        api_annotation_payload = {
            "id": 321,
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt_mixed",
                    "value": {"text": ["api mixed prompt"]},
                },
                {
                    "type": "textarea",
                    "from_name": "confidence_threshold",
                    "value": {"text": ["0.61"]},
                },
                {
                    "type": "choices",
                    "from_name": "selection_mode",
                    "value": {"choices": ["all"]},
                },
            ],
        }

        def _fake_predict(*args, **kwargs):
            captured["text_prompt"] = args[1]
            captured["text_prompt_source"] = kwargs["text_prompt_source"]
            captured["confidence_threshold"] = kwargs["confidence_threshold"]
            captured["selection_mode"] = kwargs["selection_mode"]
            return model.ModelResponse(predictions=[])

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(model, "_ls_api_get_json", return_value=api_annotation_payload), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([_make_task()], context=context)

        assert captured["text_prompt"] == "api mixed prompt"
        assert captured["text_prompt_source"] == "mixed"
        assert captured["confidence_threshold"] == pytest.approx(0.61)
        assert captured["selection_mode"] == "all"

    def test_predict_reuses_cached_mixed_prompt_when_next_context_omits_controls(self):
        import model

        backend = _backend()
        fake_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
        captured = []
        task = _make_task(task_id=77)
        first_context = {
            "annotation_id": 11,
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt_mixed",
                    "value": {"text": ["cached mixed prompt"]},
                },
                {
                    "type": "choices",
                    "from_name": "selection_mode",
                    "value": {"choices": ["all"]},
                },
                {
                    "type": "rectanglelabels",
                    "original_width": 60,
                    "original_height": 40,
                    "value": {
                        "x": 10.0,
                        "y": 10.0,
                        "width": 30.0,
                        "height": 20.0,
                        "rectanglelabels": ["Object"],
                    },
                },
            ],
        }
        second_context = {
            "annotation_id": 11,
            "result": [
                {
                    "type": "rectanglelabels",
                    "original_width": 60,
                    "original_height": 40,
                    "value": {
                        "x": 12.0,
                        "y": 12.0,
                        "width": 30.0,
                        "height": 20.0,
                        "rectanglelabels": ["Object"],
                    },
                }
            ],
        }

        def _fake_predict(*args, **kwargs):
            captured.append(
                {
                    "text_prompt": args[1],
                    "text_prompt_source": kwargs["text_prompt_source"],
                    "selection_mode": kwargs["selection_mode"],
                }
            )
            return model.ModelResponse(predictions=[])

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(model, "_runtime_controls_cache", {}), \
             patch.object(model, "_ls_api_get_json", return_value=None), \
             patch.object(backend, "_predict_sam3", side_effect=_fake_predict), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch.object(model.Image, "open") as open_mock:
            open_mock.return_value.convert.return_value = fake_img
            backend.predict([task], context=first_context)
            backend.predict([task], context=second_context)

        assert captured[0]["text_prompt"] == "cached mixed prompt"
        assert captured[0]["text_prompt_source"] == "mixed"
        assert captured[1]["text_prompt"] == "cached mixed prompt"
        assert captured[1]["text_prompt_source"] == "mixed"
        assert captured[1]["selection_mode"] == "all"


class TestRoute3ImageOutput:
    def test_text_only_adaptive_returns_multiple_masks(self):
        import model

        backend = _backend()
        processor = MagicMock()
        state = _make_state([0.93, 0.82, 0.71, 0.33])

        processor.set_image.return_value = {}
        processor.set_text_prompt.return_value = state
        processor.add_geometric_prompt.return_value = state

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(model, "_processor", processor), \
             patch.object(model, "RETURN_ALL_MASKS", False), \
             patch.object(model, "MASK_SELECTION_MODE", "adaptive"), \
             patch.object(model, "MAX_RETURNED_MASKS", 3), \
             patch.object(model, "MIN_RETURN_SCORE", 0.0):
            output = backend._predict_sam3_inner(
                image=image,
                text_prompt="car",
                point_coords=[],
                point_labels=[],
                input_boxes=[],
                selected_label="Object",
                image_width=64,
                image_height=64,
                from_name="brush",
                to_name="image",
                confidence_threshold=0.5,
                selection_mode="adaptive",
                text_prompt_source="pure",
            )

        prediction = output.predictions[0]
        dump = prediction.model_dump() if hasattr(prediction, "model_dump") else prediction
        brush_items = [item for item in dump.get("result", []) if item.get("type") == "brushlabels"]
        assert len(brush_items) == 3
        score_items = [item for item in dump.get("result", []) if item.get("type") == "textarea"]
        assert score_items
        score_text = "\n".join(score_items[0].get("value", {}).get("text", []))
        assert "mode=text_only" in score_text
        assert "selection_mode=adaptive" in score_text

    def test_top1_mode_returns_single_mask(self):
        import model

        backend = _backend()
        processor = MagicMock()
        state = _make_state([0.92, 0.84, 0.51])

        processor.set_image.return_value = {}
        processor.set_text_prompt.return_value = state
        processor.add_geometric_prompt.return_value = state

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(model, "_processor", processor), \
             patch.object(model, "RETURN_ALL_MASKS", False), \
             patch.object(model, "MASK_SELECTION_MODE", "top1"), \
             patch.object(model, "MAX_RETURNED_MASKS", 3), \
             patch.object(model, "MIN_RETURN_SCORE", 0.0):
            output = backend._predict_sam3_inner(
                image=image,
                text_prompt="car",
                point_coords=[],
                point_labels=[],
                input_boxes=[],
                selected_label="Object",
                image_width=64,
                image_height=64,
                from_name="brush",
                to_name="image",
                confidence_threshold=0.5,
                selection_mode="top1",
                text_prompt_source="pure",
            )

        prediction = output.predictions[0]
        dump = prediction.model_dump() if hasattr(prediction, "model_dump") else prediction
        brush_items = [item for item in dump.get("result", []) if item.get("type") == "brushlabels"]
        assert len(brush_items) == 1

    def test_mixed_geo_mode_is_reported_in_score_lines(self):
        import model

        backend = _backend()
        processor = MagicMock()
        state = _make_state([0.88, 0.51])

        processor.set_image.return_value = {}
        processor.set_text_prompt.return_value = state
        processor.add_geometric_prompt.return_value = state

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(model, "_processor", processor):
            output = backend._predict_sam3_inner(
                image=image,
                text_prompt="crack",
                point_coords=[],
                point_labels=[],
                input_boxes=[([6.0, 6.0, 20.0, 20.0], True)],
                selected_label="Object",
                image_width=64,
                image_height=64,
                from_name="brush",
                to_name="image",
                confidence_threshold=0.5,
                selection_mode="adaptive",
                text_prompt_source="mixed",
            )

        prediction = output.predictions[0]
        dump = prediction.model_dump() if hasattr(prediction, "model_dump") else prediction
        score_items = [item for item in dump.get("result", []) if item.get("type") == "textarea"]
        assert score_items
        score_text = "\n".join(score_items[0].get("value", {}).get("text", []))
        assert "mode=mixed_text_geo" in score_text

    def test_pure_geo_mode_is_reported_in_score_lines(self):
        import model

        backend = _backend()
        processor = MagicMock()
        state = _make_state([0.87, 0.62])

        processor.set_image.return_value = {}
        processor.set_text_prompt.return_value = state
        processor.add_geometric_prompt.return_value = state

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(model, "_processor", processor):
            output = backend._predict_sam3_inner(
                image=image,
                text_prompt="crack",
                point_coords=[],
                point_labels=[],
                input_boxes=[([6.0, 6.0, 20.0, 20.0], True)],
                selected_label="Object",
                image_width=64,
                image_height=64,
                from_name="brush",
                to_name="image",
                confidence_threshold=0.5,
                selection_mode="adaptive",
                text_prompt_source="pure_geo",
            )

        prediction = output.predictions[0]
        dump = prediction.model_dump() if hasattr(prediction, "model_dump") else prediction
        score_items = [item for item in dump.get("result", []) if item.get("type") == "textarea"]
        assert score_items
        score_text = "\n".join(score_items[0].get("value", {}).get("text", []))
        assert "mode=text_geo" in score_text
        assert "text_source=pure_geo" in score_text

    def test_threshold_mode_can_filter_geo_only_results(self):
        import model

        backend = _backend()
        processor = MagicMock()
        state = _make_state([0.84, 0.73])

        processor.set_image.return_value = {}
        processor.set_text_prompt.return_value = state
        processor.add_geometric_prompt.return_value = state

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(model, "_processor", processor):
            output = backend._predict_sam3_inner(
                image=image,
                text_prompt=None,
                point_coords=[],
                point_labels=[],
                input_boxes=[([8.0, 8.0, 24.0, 24.0], True)],
                selected_label="Object",
                image_width=64,
                image_height=64,
                from_name="brush",
                to_name="image",
                confidence_threshold=0.95,
                selection_mode="threshold",
                text_prompt_source="none",
            )

        assert output.predictions == []

    def test_top1_mode_ignores_confidence_threshold_when_global_switch_off(self):
        import model

        backend = _backend()
        processor = MagicMock()
        state = _make_state([0.84, 0.73])

        processor.set_image.return_value = {}
        processor.set_text_prompt.return_value = state
        processor.add_geometric_prompt.return_value = state

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(model, "_processor", processor), \
             patch.object(model, "APPLY_THRESHOLD_GLOBALLY", False), \
             patch.object(model, "MIN_RETURN_SCORE", 0.0):
            output = backend._predict_sam3_inner(
                image=image,
                text_prompt="crack",
                point_coords=[],
                point_labels=[],
                input_boxes=[],
                selected_label="Object",
                image_width=64,
                image_height=64,
                from_name="brush",
                to_name="image",
                confidence_threshold=0.95,
                selection_mode="top1",
                text_prompt_source="pure",
            )

        prediction = output.predictions[0]
        dump = prediction.model_dump() if hasattr(prediction, "model_dump") else prediction
        brush_items = [item for item in dump.get("result", []) if item.get("type") == "brushlabels"]
        assert len(brush_items) == 1

    def test_threshold_mode_still_uses_confidence_threshold_when_global_switch_off(self):
        import model

        backend = _backend()
        processor = MagicMock()
        state = _make_state([0.84, 0.73])

        processor.set_image.return_value = {}
        processor.set_text_prompt.return_value = state
        processor.add_geometric_prompt.return_value = state

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(model, "_processor", processor), \
             patch.object(model, "APPLY_THRESHOLD_GLOBALLY", False), \
             patch.object(model, "MIN_RETURN_SCORE", 0.0):
            output = backend._predict_sam3_inner(
                image=image,
                text_prompt="crack",
                point_coords=[],
                point_labels=[],
                input_boxes=[],
                selected_label="Object",
                image_width=64,
                image_height=64,
                from_name="brush",
                to_name="image",
                confidence_threshold=0.95,
                selection_mode="threshold",
                text_prompt_source="pure",
            )

        assert output.predictions == []

    def test_adaptive_mode_returns_empty_when_all_scores_below_threshold(self):
        import model

        backend = _backend()
        processor = MagicMock()
        state = _make_state([0.82, 0.71])

        processor.set_image.return_value = {}
        processor.set_text_prompt.return_value = state
        processor.add_geometric_prompt.return_value = state

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with patch.object(model, "_processor", processor):
            output = backend._predict_sam3_inner(
                image=image,
                text_prompt="crack",
                point_coords=[],
                point_labels=[],
                input_boxes=[],
                selected_label="Object",
                image_width=64,
                image_height=64,
                from_name="brush",
                to_name="image",
                confidence_threshold=0.95,
                selection_mode="adaptive",
                text_prompt_source="pure",
            )

        assert output.predictions == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
