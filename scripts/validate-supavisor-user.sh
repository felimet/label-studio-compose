#!/bin/bash
set -euo pipefail

SUPABASE_ENV_FILE="${1:-.env.supabase}"

if [ ! -f .env ]; then
  echo "Missing .env" >&2
  exit 1
fi

if [ ! -f "$SUPABASE_ENV_FILE" ]; then
  echo "Missing $SUPABASE_ENV_FILE" >&2
  exit 1
fi

host=$(grep -E '^LS_POSTGRE_HOST=' .env | cut -d= -f2- | tr -d '[:space:]' || true)
user=$(grep -E '^LS_POSTGRE_USER=' .env | cut -d= -f2- | tr -d '[:space:]' || true)
tenant=$(grep -E '^POOLER_TENANT_ID=' "$SUPABASE_ENV_FILE" | cut -d= -f2- | tr -d '[:space:]' || true)

if [ "$host" != "supavisor" ]; then
  exit 0
fi

case "$user" in
  *.*) ;;
  *)
    echo "Invalid LS_POSTGRE_USER: when LS_POSTGRE_HOST=supavisor, use postgres.<POOLER_TENANT_ID>." >&2
    exit 1
    ;;
esac

if [ -n "$tenant" ] && [ "$user" != "postgres.$tenant" ]; then
  echo "Invalid LS_POSTGRE_USER: expected postgres.$tenant (from $SUPABASE_ENV_FILE POOLER_TENANT_ID)." >&2
  exit 1
fi
