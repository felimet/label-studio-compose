"""Tests for context_builder.py."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.utils.context_builder import (
    build_sam3_text_context,
    build_sam21_grid_context,
)


def test_sam3_context_format():
    ctx = build_sam3_text_context("cow, sheep")
    items = ctx["result"]
    text_items = [i for i in items if i["from_name"] == "text_prompt"]
    assert len(text_items) == 1
    assert text_items[0]["type"] == "textarea"
    assert text_items[0]["to_name"] == "image"
    assert "cow" in text_items[0]["value"]["text"][0]
    assert "sheep" in text_items[0]["value"]["text"][0]


def test_sam3_confidence_injection():
    ctx = build_sam3_text_context("cow", confidence=0.7)
    items = ctx["result"]
    threshold_items = [i for i in items if i["from_name"] == "confidence_threshold"]
    assert len(threshold_items) == 1
    assert threshold_items[0]["value"]["text"][0] == "0.7"
    assert threshold_items[0]["type"] == "textarea"


def test_sam3_default_confidence_omitted():
    ctx = build_sam3_text_context("cow", confidence=0.5)
    from_names = [i["from_name"] for i in ctx["result"]]
    assert "confidence_threshold" not in from_names


def test_sam21_grid_shape():
    ctx = build_sam21_grid_context(["pig"], grid_n=3)
    kp_items = [i for i in ctx["result"] if i["type"] == "keypointlabels"]
    assert len(kp_items) == 9  # 3×3


def test_sam21_grid_coordinates():
    ctx = build_sam21_grid_context(["pig"], grid_n=3)
    kp_items = [i for i in ctx["result"] if i["type"] == "keypointlabels"]
    for item in kp_items:
        x = item["value"]["x"]
        y = item["value"]["y"]
        assert 0 < x < 100, f"x={x} out of (0,100)"
        assert 0 < y < 100, f"y={y} out of (0,100)"


def test_sam21_grid_has_brush_hint():
    ctx = build_sam21_grid_context(["cow"], grid_n=2)
    brush_items = [i for i in ctx["result"] if i["type"] == "brushlabels"]
    assert len(brush_items) == 1
    assert brush_items[0]["value"]["brushlabels"] == ["cow"]
