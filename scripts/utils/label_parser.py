from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List


def extract_label_names(xml_config: str) -> List[str]:
    """Parse Label Studio labeling config XML and return all <Label value> strings."""
    root = ET.fromstring(xml_config)
    return [
        label.get("value", "")
        for label in root.iter("Label")
        if label.get("value")
    ]


def has_brush_labels(xml_config: str) -> bool:
    """Return True if the labeling config contains a <BrushLabels> tag."""
    root = ET.fromstring(xml_config)
    return any(True for _ in root.iter("BrushLabels"))
