#!/bin/bash
# Full stack health check — validates all services are responsive.
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILED=1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-label-anything-sam}"
SUPABASE_ENV_FILE="${SUPABASE_STANDALONE_ENV:-.env.supabase}"

core_compose() {
    docker compose --project-name "$PROJECT_NAME" "$@"
}

supabase_compose() {
    docker compose \
        --project-name "$PROJECT_NAME" \
        --env-file .env \
        --env-file "$SUPABASE_ENV_FILE" \
        -f docker-compose.yml \
        -f docker-compose.supabase.yml \
        "$@"
}

FAILED=0

echo "═══════════════════════════════════"
echo "  Label Studio Stack Health Check"
echo "═══════════════════════════════════"

# ── Supabase DB / Pooler ───────────────────────────────────
echo ""
echo "── Supabase DB / Supavisor ──"
if supabase_compose exec -T db pg_isready -U postgres -h localhost >/dev/null 2>&1; then
    pass "Supabase db accepting connections"
else
    fail "Supabase db not ready"
fi

if supabase_compose exec -T supavisor curl -sSfL --head -o /dev/null http://127.0.0.1:4000/api/health >/dev/null 2>&1; then
    pass "Supavisor /api/health OK"
else
    fail "Supavisor not ready"
fi

# ── Redis ───────────────────────────────────────────────────
echo ""
echo "── Redis ──"
REDIS_PASS=$(grep '^REDIS_PASSWORD=' .env 2>/dev/null | cut -d= -f2 || echo "")
if core_compose exec -T redis redis-cli -a "$REDIS_PASS" ping 2>/dev/null | grep -q PONG; then
    pass "Redis PONG"
else
    fail "Redis not responding"
fi

# ── MinIO ───────────────────────────────────────────────────
echo ""
echo "── MinIO ──"
if core_compose exec -T minio mc ready local 2>/dev/null; then
    pass "MinIO API healthy"
else
    fail "MinIO not ready"
fi

BUCKET=$(grep '^MINIO_BUCKET=' .env 2>/dev/null | cut -d= -f2 || echo "label-studio-bucket")
if core_compose exec -T minio mc ls "local/${BUCKET}" 2>/dev/null; then
    pass "MinIO bucket '${BUCKET}' accessible"
else
    warn "MinIO bucket '${BUCKET}' not found — run: make init-minio"
fi

# ── Label Studio ────────────────────────────────────────────
echo ""
echo "── Label Studio ──"
if core_compose exec -T label-studio curl -sf http://localhost:8080/health 2>/dev/null; then
    pass "Label Studio /health OK"
else
    fail "Label Studio not responding"
fi

# ── Nginx ───────────────────────────────────────────────────
echo ""
echo "── Nginx ──"
if core_compose exec -T nginx curl -sf http://localhost/health 2>/dev/null | grep -q OK; then
    pass "Nginx proxy healthy"
else
    fail "Nginx not responding"
fi

# ── SAM3 ML Backends (optional — only when ML stack running) ─
echo ""
echo "── SAM3 ML Backends ──"
for svc in sam3-image-backend sam3-video-backend; do
    if docker compose --project-name "$PROJECT_NAME" -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml ps "$svc" 2>/dev/null | grep -qiE "Up|running"; then
        if docker compose --project-name "$PROJECT_NAME" -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml exec -T "$svc" curl -sf http://localhost:9090/health 2>/dev/null; then
            pass "$svc /health OK"
        else
            fail "$svc not responding"
        fi
    else
        warn "$svc not running (start with: make ml-up)"
    fi
done

# ── Cloudflared ─────────────────────────────────────────────
echo ""
echo "── Cloudflare Tunnel ──"
if core_compose ps cloudflared 2>/dev/null | grep -qiE "Up|running"; then
    pass "cloudflared container running"
else
    warn "cloudflared not running"
fi

echo ""
echo "═══════════════════════════════════"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}All checks passed.${NC}"
else
    echo -e "${RED}${FAILED} check(s) failed — see above.${NC}"
    exit 1
fi
