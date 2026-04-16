#!/bin/bash
set -euo pipefail

SUPABASE_ENV_FILE="${1:-.env.supabase}"

if [ ! -f "$SUPABASE_ENV_FILE" ]; then
  echo "Missing $SUPABASE_ENV_FILE" >&2
  exit 1
fi

schemas=$(grep -E '^PGRST_DB_SCHEMAS=' "$SUPABASE_ENV_FILE" | cut -d= -f2- | tr -d '[:space:]' || true)

# Empty value is acceptable because compose fallback enforces a safe default.
if [ -z "$schemas" ]; then
  exit 0
fi

if [[ "$schemas" == *"public"* ]]; then
  if [ "${ALLOW_POSTGREST_PUBLIC_SCHEMA_EXPOSURE:-false}" != "true" ]; then
    echo "Unsafe PostgREST exposure detected in $SUPABASE_ENV_FILE: PGRST_DB_SCHEMAS=$schemas" >&2
    echo "For high security, remove public from PGRST_DB_SCHEMAS (recommended: storage)." >&2
    echo "If this is intentional, set ALLOW_POSTGREST_PUBLIC_SCHEMA_EXPOSURE=true when running make." >&2
    exit 1
  fi
fi
