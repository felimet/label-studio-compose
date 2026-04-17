"""Route-3 regression tests for SAM3 video backend."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("SAM3_MODEL_ID", "test/model")
os.environ.setdefault("SAM3_CHECKPOINT_FILENAME", "model.pt")
os.environ.setdefault("MODEL_DIR", "/tmp/test-models")
os.environ.setdefault("MAX_FRAMES_TO_TRACK", "3")
os.environ.setdefault("SAM3_ENABLE_PCS", "true")
os.environ.setdefault("SAM3_ENABLE_FA3", "false")
os.environ.setdefault("SAM3_ENABLE_BIDIRECTIONAL_TRACKING", "true")


_LABEL_CONFIG = """
<View>
  <Labels name="videoLabels" toName="video" allowEmpty="true">
    <Label value="Object"/>
    <Label value="Exclude"/>
  </Labels>
  <Video name="video" value="$video" framerate="25.0"/>
  <VideoRectangle name="box" toName="video" smart="true"/>
    <TextArea name="text_prompt" toName="video" maxSubmissions="1" editable="true"/>
    <TextArea name="text_prompt_mixed" toName="video" maxSubmissions="1" editable="true"/>
</View>
"""


def _make_task(url="/tmp/fake.mp4", task_id=1):
    return {"id": task_id, "data": {"video": url}}


def _make_vr_context(obj_id="obj-1", frame=1):
    return {
        "result": [
            {
                "type": "videorectangle",
                "id": obj_id,
                "value": {
                    "framesCount": 20,
                    "duration": 1.0,
                    "labels": ["Object"],
                    "sequence": [
                        {
                            "frame": frame,
                            "enabled": True,
                            "x": 10.0,
                            "y": 10.0,
                            "width": 30.0,
                            "height": 20.0,
                        }
                    ],
                },
            }
        ]
    }


def _make_mock_predictor(stream_payload):
    predictor = MagicMock()

    def _dispatch_request(request):
        req_type = request["type"]
        if req_type == "start_session":
            return {"session_id": "sess-1"}
        if req_type == "add_prompt":
            return {"accepted": True, "frame_index": request.get("frame_index", 0)}
        if req_type == "close_session":
            return {"closed": True}
        return {}

    def _dispatch_stream(_request):
        for item in stream_payload:
            yield item

    predictor.handle_request.side_effect = _dispatch_request
    predictor.handle_stream_request.side_effect = _dispatch_stream
    return predictor


class _FakeCapture:
    def __init__(self, *_args, **_kwargs):
        pass

    def get(self, prop):
        import model

        if prop == model.cv2.CAP_PROP_FRAME_WIDTH:
            return 320
        if prop == model.cv2.CAP_PROP_FRAME_HEIGHT:
            return 180
        if prop == model.cv2.CAP_PROP_FRAME_COUNT:
            return 30
        return 0

    def release(self):
        return None


def _backend():
    import model

    return model.NewModel(project_id="test", label_config=_LABEL_CONFIG)


def _raw_backend_instance():
    import model

    return object.__new__(model.NewModel)


class TestPromptParsing:
    def test_get_text_prompt_skips_scores_textarea(self):
        import model

        backend = _raw_backend_instance()
        ctx = {
            "result": [
                {"type": "textarea", "from_name": "scores", "value": {"text": ["#0 score=0.99"]}},
                {"type": "textarea", "from_name": "text_prompt_mixed", "value": {"text": ["person"]}},
            ]
        }
        assert backend._get_text_prompt(ctx) == ("person", None)

    def test_get_text_prompt_parses_legacy_field(self):
        backend = _raw_backend_instance()
        ctx = {
            "result": [
                {"type": "textarea", "from_name": "text_prompt", "value": {"text": ["car"]}},
            ]
        }
        assert backend._get_text_prompt(ctx) == (None, "car")

    def test_get_geo_prompts_parses_videorect_and_keypoint(self):
        backend = _raw_backend_instance()
        ctx = {
            "result": [
                {
                    "type": "videorectangle",
                    "id": "obj-a",
                    "value": {
                        "labels": ["Object"],
                        "sequence": [
                            {
                                "frame": 3,
                                "enabled": True,
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                            }
                        ],
                    },
                },
                {
                    "type": "keypointlabels",
                    "id": "obj-a",
                    "value": {"frame": 3, "x": 55.0, "y": 45.0, "keypointlabels": ["background"]},
                    "is_positive": 0,
                },
            ]
        }
        prompts = backend._get_geo_prompts(ctx, default_label="Object")
        assert len(prompts) == 2
        assert prompts[0]["frame_idx"] == 2
        assert prompts[1]["type"] == "point"

    def test_keypoint_binds_to_single_track_on_same_frame(self):
        backend = _raw_backend_instance()
        ctx = {
            "result": [
                {
                    "type": "videorectangle",
                    "id": "track-1",
                    "value": {
                        "labels": ["Object"],
                        "sequence": [
                            {
                                "frame": 2,
                                "enabled": True,
                                "x": 10.0,
                                "y": 10.0,
                                "width": 20.0,
                                "height": 20.0,
                            }
                        ],
                    },
                },
                {
                    "type": "keypointlabels",
                    "id": "kp-free",
                    "value": {"frame": 2, "x": 20.0, "y": 20.0, "keypointlabels": ["Object"]},
                },
            ]
        }

        prompts = backend._get_geo_prompts(ctx, default_label="Object")
        point_prompts = [p for p in prompts if p["type"] == "point"]
        assert len(point_prompts) == 1
        assert point_prompts[0]["obj_id"] == "track-1"


class TestRoute3VideoCore:
    def test_point_prompt_uses_native_points_payload(self):
        import model

        stream_payload = []
        predictor = _make_mock_predictor(stream_payload)
        backend = _raw_backend_instance()

        geo_prompts = [
            {
                "type": "point",
                "obj_id": "obj-point",
                "frame_idx": 5,
                "x_pct": 50.0,
                "y_pct": 50.0,
                "is_positive": True,
            }
        ]

        with patch.object(model, "_predictor", predictor), \
             patch.object(backend, "_extract_frames", return_value=4), \
             patch.object(model.cv2, "VideoCapture", return_value=_FakeCapture()):
            tracked, _ = backend._predict_sam3_inner("/tmp/fake.mp4", geo_prompts, None, 25.0)

        assert tracked == {}
        add_prompt_requests = [
            c.args[0]
            for c in predictor.handle_request.call_args_list
            if c.args[0].get("type") == "add_prompt"
        ]
        assert add_prompt_requests
        point_requests = [req for req in add_prompt_requests if "points" in req]
        assert point_requests
        assert all("bounding_boxes" not in req for req in point_requests)

    def test_bidirectional_propagation_is_requested(self):
        import model

        stream_payload = []
        predictor = _make_mock_predictor(stream_payload)
        backend = _raw_backend_instance()

        geo_prompts = [
            {
                "type": "box",
                "obj_id": "obj-box",
                "frame_idx": 2,
                "x_pct": 10.0,
                "y_pct": 10.0,
                "w_pct": 20.0,
                "h_pct": 20.0,
                "is_positive": True,
            }
        ]

        with patch.object(model, "_predictor", predictor), \
             patch.object(backend, "_extract_frames", return_value=4), \
             patch.object(model.cv2, "VideoCapture", return_value=_FakeCapture()):
            backend._predict_sam3_inner("/tmp/fake.mp4", geo_prompts, None, 25.0)

        directions = [
            c.args[0].get("propagation_direction")
            for c in predictor.handle_stream_request.call_args_list
        ]
        assert "forward" in directions
        assert "reverse" in directions or "backward" in directions

    def test_tracking_results_are_split_per_object(self):
        import model

        mask_a = np.zeros((20, 20), dtype=bool)
        mask_b = np.zeros((20, 20), dtype=bool)
        mask_a[2:8, 2:8] = True
        mask_b[10:16, 10:16] = True

        stream_payload = [
            {
                "frame_index": 1,
                "outputs": {
                    "out_obj_ids": np.array([0, 1]),
                    "out_binary_masks": np.stack([mask_a, mask_b]),
                },
            }
        ]
        predictor = _make_mock_predictor(stream_payload)
        backend = _raw_backend_instance()

        geo_prompts = [
            {
                "type": "box",
                "obj_id": "a",
                "frame_idx": 0,
                "x_pct": 10.0,
                "y_pct": 10.0,
                "w_pct": 20.0,
                "h_pct": 20.0,
                "is_positive": True,
            },
            {
                "type": "box",
                "obj_id": "b",
                "frame_idx": 0,
                "x_pct": 50.0,
                "y_pct": 50.0,
                "w_pct": 20.0,
                "h_pct": 20.0,
                "is_positive": True,
            },
        ]

        with patch.object(model, "_predictor", predictor), \
             patch.object(backend, "_extract_frames", return_value=4), \
             patch.object(model.cv2, "VideoCapture", return_value=_FakeCapture()):
            tracked, _ = backend._predict_sam3_inner("/tmp/fake.mp4", geo_prompts, None, 25.0)

        assert set(tracked.keys()) == {"a", "b"}
        assert tracked["a"]
        assert tracked["b"]

    def test_forward_stream_error_is_tolerated(self):
        import model

        predictor = MagicMock()

        def _dispatch_request(request):
            if request["type"] == "start_session":
                return {"session_id": "sess-1"}
            if request["type"] in {"add_prompt", "close_session"}:
                return {"ok": True}
            return {}

        def _dispatch_stream(request):
            if request.get("propagation_direction") == "forward":
                raise RuntimeError("forward failed")
            return iter(())

        predictor.handle_request.side_effect = _dispatch_request
        predictor.handle_stream_request.side_effect = _dispatch_stream

        geo_prompts = [
            {
                "type": "box",
                "obj_id": "obj-box",
                "frame_idx": 2,
                "x_pct": 10.0,
                "y_pct": 10.0,
                "w_pct": 20.0,
                "h_pct": 20.0,
                "is_positive": True,
            }
        ]
        backend = _raw_backend_instance()

        with patch.object(model, "_predictor", predictor), \
             patch.object(backend, "_extract_frames", return_value=4), \
             patch.object(model.cv2, "VideoCapture", return_value=_FakeCapture()):
            tracked, score_lines = backend._predict_sam3_inner("/tmp/fake.mp4", geo_prompts, None, 25.0)

        assert tracked == {}
        assert isinstance(score_lines, list)
        close_calls = [
            c.args[0]
            for c in predictor.handle_request.call_args_list
            if c.args[0].get("type") == "close_session"
        ]
        assert close_calls


class TestPredictOutputMerge:
    def test_mixed_text_without_geo_does_not_trigger_inference(self):
        import model

        backend = _backend()
        context = {
            "result": [
                {
                    "type": "textarea",
                    "from_name": "text_prompt_mixed",
                    "value": {"text": ["person"]},
                }
            ]
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "_predict_sam3") as predict_mock:
            result = backend.predict([_make_task()], context=context)

        assert result.predictions == []
        predict_mock.assert_not_called()

    def test_predict_uses_mixed_text_geo_mode_when_box_prompt_exists(self):
        import model

        backend = _backend()
        context = _make_vr_context(obj_id="obj-1", frame=1)
        context["result"].append(
            {
                "type": "textarea",
                "from_name": "text_prompt_mixed",
                "value": {"text": ["person"]},
            }
        )

        tracked_sequences = {
            "obj-1": [
                {
                    "frame": 2,
                    "x": 12.0,
                    "y": 12.0,
                    "width": 28.0,
                    "height": 18.0,
                    "enabled": True,
                    "rotation": 0,
                    "time": 0.04,
                }
            ]
        }
        geo_prompts_stub = [
            {
                "type": "box",
                "obj_id": "obj-1",
                "frame_idx": 0,
                "x_pct": 10.0,
                "y_pct": 10.0,
                "w_pct": 20.0,
                "h_pct": 20.0,
                "is_positive": True,
            }
        ]

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"), \
             patch.object(backend, "_get_geo_prompts", return_value=geo_prompts_stub), \
             patch.object(backend, "_get_text_prompt", return_value=("person", None)), \
             patch.object(backend, "_predict_sam3", return_value=(tracked_sequences, ["ok"])) as predict_mock:
            result = backend.predict([_make_task()], context=context)

        assert predict_mock.call_args.args[2] == "person"
        pred = result.predictions[0]
        dump = pred.model_dump() if hasattr(pred, "model_dump") else pred
        text_items = [item for item in dump.get("result", []) if item.get("type") == "textarea"]
        assert text_items
        score_text = "\n".join(text_items[0].get("value", {}).get("text", []))
        assert "mode=mixed_text_geo" in score_text

    def test_predict_downgrades_to_geo_only_mode_for_point_prompt_with_text(self):
        import model

        backend = _backend()
        context = _make_vr_context(obj_id="obj-1", frame=1)
        context["result"].append(
            {
                "type": "textarea",
                "from_name": "text_prompt_mixed",
                "value": {"text": ["person"]},
            }
        )

        tracked_sequences = {
            "obj-1": [
                {
                    "frame": 2,
                    "x": 12.0,
                    "y": 12.0,
                    "width": 28.0,
                    "height": 18.0,
                    "enabled": True,
                    "rotation": 0,
                    "time": 0.04,
                }
            ]
        }
        point_only_geo_prompts = [
            {
                "type": "point",
                "obj_id": "obj-1",
                "frame_idx": 0,
                "x_pct": 50.0,
                "y_pct": 50.0,
                "is_positive": True,
            }
        ]

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"), \
             patch.object(backend, "_get_geo_prompts", return_value=point_only_geo_prompts), \
             patch.object(backend, "_get_text_prompt", return_value=("person", None)), \
             patch.object(backend, "_predict_sam3", return_value=(tracked_sequences, ["ok"])) as predict_mock:
            result = backend.predict([_make_task()], context=context)

        assert predict_mock.call_args.args[2] is None
        pred = result.predictions[0]
        dump = pred.model_dump() if hasattr(pred, "model_dump") else pred
        text_items = [item for item in dump.get("result", []) if item.get("type") == "textarea"]
        assert text_items
        score_text = "\n".join(text_items[0].get("value", {}).get("text", []))
        assert "mode=geo_only" in score_text

    def test_predict_merges_existing_and_new_sequences_per_object(self):
        import model

        backend = _backend()
        context = _make_vr_context(obj_id="obj-1", frame=1)

        tracked_sequences = {
            "obj-1": [
                {
                    "frame": 2,
                    "x": 12.0,
                    "y": 12.0,
                    "width": 28.0,
                    "height": 18.0,
                    "enabled": True,
                    "rotation": 0,
                    "time": 0.04,
                }
            ],
            "obj-2": [
                {
                    "frame": 2,
                    "x": 60.0,
                    "y": 30.0,
                    "width": 20.0,
                    "height": 20.0,
                    "enabled": True,
                    "rotation": 0,
                    "time": 0.04,
                }
            ],
        }

        with patch.object(model, "_ensure_loaded", return_value=None), \
             patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"), \
             patch.object(backend, "_predict_sam3", return_value=(tracked_sequences, ["ok"])):
            result = backend.predict([_make_task()], context=context)

        assert result.predictions
        pred = result.predictions[0]
        dump = pred.model_dump() if hasattr(pred, "model_dump") else pred

        rects = [item for item in dump.get("result", []) if item.get("type") == "videorectangle"]
        assert len(rects) == 2

        by_id = {item["id"]: item for item in rects}
        assert "obj-1" in by_id and "obj-2" in by_id

        seq_obj1 = by_id["obj-1"]["value"]["sequence"]
        frames_obj1 = [s["frame"] for s in seq_obj1]
        assert 1 in frames_obj1 and 2 in frames_obj1

        text_items = [item for item in dump.get("result", []) if item.get("type") == "textarea"]
        assert text_items
        score_text = "\n".join(text_items[0].get("value", {}).get("text", []))
        assert "mode=geo_only" in score_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
