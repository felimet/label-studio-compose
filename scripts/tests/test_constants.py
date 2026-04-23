"""Tests for constants.py — includes grep guard for inline literals."""
from __future__ import annotations

import pathlib
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.utils.constants import (
    CLI_MODEL_VERSION_SAM21,
    CLI_MODEL_VERSION_SAM3,
    SAM3_MIXED_FROM_NAME,
    SAM3_PURE_TEXT_FROM_NAME,
)


def test_pure_text_from_name_value():
    assert SAM3_PURE_TEXT_FROM_NAME == "text_prompt"


def test_mixed_from_name_value():
    assert SAM3_MIXED_FROM_NAME == "text_prompt_mixed"


def test_model_version_sam3():
    assert CLI_MODEL_VERSION_SAM3 == "batch-annotate-sam3-v1"


def test_model_version_sam21():
    assert CLI_MODEL_VERSION_SAM21 == "batch-annotate-sam21-v1"


def test_no_inline_text_prompt_mixed():
    """SAM3_MIXED_FROM_NAME value must not appear as an inline literal outside constants.py.

    Excluded: constants.py (definition) and test_*.py files (allowed for test assertions).
    """
    scripts_root = pathlib.Path(__file__).parent.parent
    py_files = [
        p for p in scripts_root.rglob("*.py")
        if p.name != "constants.py" and not p.name.startswith("test_")
    ]
    violations = []
    for f in py_files:
        if SAM3_MIXED_FROM_NAME in f.read_text(encoding="utf-8"):
            violations.append(str(f))
    assert not violations, (
        f"Inline mixed from_name literal found in {violations}; "
        "use SAM3_MIXED_FROM_NAME from constants.py"
    )
