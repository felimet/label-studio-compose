from __future__ import annotations

from .constants import SAM3_PURE_TEXT_FROM_NAME


def build_sam3_text_context(
    label_names: list[str],
    confidence: float = 0.5,
) -> dict:
    """Build SAM3 pure-text context for batch annotation.

    from_name MUST be SAM3_PURE_TEXT_FROM_NAME ('text_prompt').
    Never use SAM3_MIXED_FROM_NAME — that route requires co-existing geometry.
    confidence is injected as a separate textarea with from_name='confidence_threshold'
    (sam3-image/model.py:495-508); only appended when confidence != 0.5 to
    avoid redundant context items (backend default == 0.5).

    MUST NOT be called for --backend sam21. SAM2.1 has no text-prompt path.
    """
    results: list[dict] = [
        {
            "type": "textarea",
            "from_name": SAM3_PURE_TEXT_FROM_NAME,
            "to_name": "image",
            "value": {"text": [", ".join(label_names)]},
        }
    ]
    if confidence != 0.5:
        results.append(
            {
                "type": "textarea",
                "from_name": "confidence_threshold",
                "to_name": "image",
                "value": {"text": [str(confidence)]},
            }
        )
    return {"result": results}


def build_sam21_grid_context(
    label_names: list[str],
    brush_from_name: str = "tag",
    keypoint_from_name: str = "keypoint",
    to_name: str = "image",
    grid_n: int = 3,
) -> dict:
    """Synthesise NxN center-point grid as keypointlabels for SAM2.1 batch mode.

    Includes a brushlabels hint so _resolve_brush_output picks the correct
    label instead of falling back to the synthetic '_grid_point' string.
    SAM2.1 returns exactly ONE mask per call (argmax of scores, model.py:522).
    Grid mode is EXPERIMENTAL: suitable for single-dominant-object images only.
    """
    step = 100.0 / (grid_n + 1)
    offsets = [step * (i + 1) for i in range(grid_n)]
    results: list[dict] = []

    # Label hint: give _resolve_brush_output a real label to attach to the mask
    if label_names:
        results.append(
            {
                "type": "brushlabels",
                "from_name": brush_from_name,
                "to_name": to_name,
                "value": {"brushlabels": [label_names[0]]},
            }
        )

    # NxN grid of foreground points
    for x_pct in offsets:
        for y_pct in offsets:
            results.append(
                {
                    "type": "keypointlabels",
                    "from_name": keypoint_from_name,
                    "to_name": to_name,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "keypointlabels": ["_grid_point"],
                    },
                }
            )

    return {"result": results}


def build_context(backend: str, label_names: list[str], args) -> dict:
    """Dispatch to the correct context builder based on --backend.

    build_sam3_text_context() MUST NOT be called for backend=='sam21'.
    --backend sam21 without --sam21-mode grid is caught at pre-flight (exit 3).
    """
    if backend == "sam3":
        return build_sam3_text_context(label_names, confidence=args.confidence)
    elif backend == "sam21" and getattr(args, "sam21_mode", None) == "grid":
        return build_sam21_grid_context(label_names, grid_n=args.grid_n)
    else:
        raise AssertionError(
            "unreachable: sam21 without --sam21-mode grid must exit at pre-flight (exit 3)"
        )


def estimate_local_vram_gb() -> float:
    """Return approximate local GPU VRAM in GB for advisory display only.

    NOTE: This measures CLIENT-SIDE VRAM, not the ML backend server's VRAM.
    Do NOT use this value to automatically set concurrency — the backend
    server may be on a different machine. Print as a hint only.
    Returns 0.0 if torch/CUDA is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return round(total / (1024**3), 1)
    except Exception:
        pass
    return 0.0
