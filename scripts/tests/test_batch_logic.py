"""Tests for core batch annotation logic."""
from __future__ import annotations

import sys, os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.batch_annotate import (
    call_predict,
    safe_write_prediction,
)
from scripts.utils.constants import CLI_MODEL_VERSION_SAM3


# ── call_predict tests ────────────────────────────────────────────────────


def _make_response(status_code: int, json_data: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


def test_classify_protocol_fail_http_error():
    resp = _make_response(500, {})
    with patch("requests.post", return_value=resp):
        status, result, score = call_predict(
            "http://localhost:9090",
            {"id": 1, "data": {}},
            {"result": []},
        )
    assert status == "protocol_fail"
    assert result == []
    assert score == 0.0


def test_classify_zero_match():
    resp = _make_response(200, {"results": []})
    with patch("requests.post", return_value=resp):
        status, result, score = call_predict(
            "http://localhost:9090",
            {"id": 1, "data": {}},
            {"result": []},
        )
    assert status == "zero_match"


def test_classify_success():
    resp = _make_response(
        200,
        {"results": [{"result": [{"type": "brushlabels"}], "score": 0.92}]},
    )
    with patch("requests.post", return_value=resp):
        status, result, score = call_predict(
            "http://localhost:9090",
            {"id": 1, "data": {}},
            {"result": []},
        )
    assert status == "success"
    assert score == 0.92
    assert result == [{"type": "brushlabels"}]


def test_classify_network_error():
    with patch("requests.post", side_effect=ConnectionError("refused")):
        status, result, score = call_predict(
            "http://localhost:9090",
            {"id": 1, "data": {}},
            {"result": []},
        )
    assert status == "protocol_fail"


# ── safe_write_prediction tests ───────────────────────────────────────────


def _make_ls(total_annotations: int = 0, get_task_raises: bool = False):
    ls = MagicMock()
    if get_task_raises:
        ls.get_task.side_effect = Exception("network error")
    else:
        ls.get_task.return_value = {"id": 1, "total_annotations": total_annotations}
    ls.delete_cli_predictions.return_value = 0
    ls.create_prediction.return_value = {}
    return ls


def test_safe_write_race_condition():
    ls = _make_ls(total_annotations=1)
    status = safe_write_prediction(
        1, [], 0.9, ls, CLI_MODEL_VERSION_SAM3, force=False
    )
    assert status == "skip_race"
    ls.create_prediction.assert_not_called()


def test_safe_write_force_bypass():
    ls = _make_ls(total_annotations=1)
    status = safe_write_prediction(
        1, [], 0.9, ls, CLI_MODEL_VERSION_SAM3, force=True
    )
    assert status == "success"
    ls.create_prediction.assert_called_once()
    ls.delete_cli_predictions.assert_called_once()


def test_safe_write_success():
    ls = _make_ls(total_annotations=0)
    status = safe_write_prediction(
        1, [{"type": "brushlabels"}], 0.9, ls, CLI_MODEL_VERSION_SAM3
    )
    assert status == "success"
    ls.delete_cli_predictions.assert_called_once_with(1, model_version=CLI_MODEL_VERSION_SAM3)
    ls.create_prediction.assert_called_once()


def test_safe_write_error_fetch():
    ls = _make_ls(get_task_raises=True)
    status = safe_write_prediction(
        1, [], 0.9, ls, CLI_MODEL_VERSION_SAM3
    )
    assert status == "error_fetch"
    ls.delete_cli_predictions.assert_not_called()
    ls.create_prediction.assert_not_called()


# ── Exit code tests ───────────────────────────────────────────────────────


def test_missing_api_key_exits_3(monkeypatch, capsys):
    monkeypatch.delenv("LABEL_STUDIO_API_KEY", raising=False)
    from scripts.batch_annotate import load_api_key
    try:
        load_api_key()
    except SystemExit as e:
        assert e.code == 3
    else:
        raise AssertionError("Expected SystemExit(3)")
