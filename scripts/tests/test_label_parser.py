"""Tests for label_parser.py."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.utils.label_parser import extract_label_names, has_brush_labels

SAMPLE_CONFIG = """
<View>
  <BrushLabels name="tag" toName="image">
    <Label value="cow" background="#FF0000"/>
    <Label value="sheep" background="#00FF00"/>
    <Label value="person"/>
  </BrushLabels>
  <Image name="image" value="$image"/>
</View>
"""

NO_BRUSH_CONFIG = """
<View>
  <RectangleLabels name="rect" toName="image">
    <Label value="dog"/>
  </RectangleLabels>
  <Image name="image" value="$image"/>
</View>
"""


def test_extract_labels():
    names = extract_label_names(SAMPLE_CONFIG)
    assert names == ["cow", "sheep", "person"]


def test_empty_config():
    names = extract_label_names("<View></View>")
    assert names == []


def test_has_brush_labels_true():
    assert has_brush_labels(SAMPLE_CONFIG) is True


def test_has_brush_labels_false():
    assert has_brush_labels(NO_BRUSH_CONFIG) is False
