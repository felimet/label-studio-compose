# ── SAM3 from_name routing (sam3-image/model.py:558-569) ──────────────────
# Routes to pure_text_prompt path (no geometry in context result)
SAM3_PURE_TEXT_FROM_NAME = "text_prompt"
# NOT USED by this CLI; defined here to prevent inline literals elsewhere
SAM3_MIXED_FROM_NAME = "text_prompt_mixed"

# ── CLI model_version identity tags ──────────────────────────────────────
# Used to identify and scope predictions created by this CLI.
# SAM2.1 backend never sets model_version itself (sam21-image/model.py);
# CLI must pass this explicitly to create_prediction() and delete_cli_predictions().
CLI_MODEL_VERSION_SAM3 = "batch-annotate-sam3-v1"
CLI_MODEL_VERSION_SAM21 = "batch-annotate-sam21-v1"
