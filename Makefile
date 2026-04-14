.PHONY: up down restart logs ps \
        ml-up ml-down \
        up-sam3-image up-sam3-video \
        up-sam21-image up-sam21-video \
        restart-sam3-image restart-sam3-video \
        restart-sam21-image restart-sam21-video \
        build-sam3-image build-sam3-video \
        build-sam21-image build-sam21-video \
        test-sam3-image test-sam3-video \
        test-sam21-image test-sam21-video \
		supabase-up supabase-down supabase-logs \
		supabase-s3-up supabase-s3-down supabase-s3-logs \
	tools-up tools-down tools-logs \
        init-minio health create-admin reset-password \
        push

# ─── Core stack ─────────────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

# ─── ML Backends (SAM3 + SAM2.1 image + video) ──────────────
# override.yml must be included explicitly when using -f flags
# (Docker Compose only auto-loads override.yml when no -f is specified)
ML_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml
SUPABASE_COMPOSE_BASE = docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.supabase.yml
SUPABASE_COMPOSE = docker compose --env-file .env --env-file .env.supabase -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.supabase.yml
SUPABASE_S3_COMPOSE = docker compose --profile supabase-s3 --env-file .env --env-file .env.supabase -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.supabase.yml
TOOLS_COMPOSE_BASE = docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.tools.yml
TOOLS_COMPOSE = docker compose --env-file .env --env-file .env.tools -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.tools.yml

ml-up:
	$(ML_COMPOSE) up -d

ml-down:
	$(ML_COMPOSE) down

up-sam3-image:
	$(ML_COMPOSE) up -d --no-deps sam3-image-backend

up-sam3-video:
	$(ML_COMPOSE) up -d --no-deps sam3-video-backend

up-sam21-image:
	$(ML_COMPOSE) up -d --no-deps sam21-image-backend

up-sam21-video:
	$(ML_COMPOSE) up -d --no-deps sam21-video-backend

restart-sam3-image:
	$(ML_COMPOSE) restart sam3-image-backend

restart-sam3-video:
	$(ML_COMPOSE) restart sam3-video-backend

restart-sam21-image:
	$(ML_COMPOSE) restart sam21-image-backend

restart-sam21-video:
	$(ML_COMPOSE) restart sam21-video-backend

build-sam3-image:
	$(ML_COMPOSE) build sam3-image-backend

build-sam3-video:
	$(ML_COMPOSE) build sam3-video-backend

build-sam21-image:
	$(ML_COMPOSE) build sam21-image-backend

build-sam21-video:
	$(ML_COMPOSE) build sam21-video-backend

test-sam3-image:
	$(ML_COMPOSE) exec sam3-image-backend python -m pytest tests/ --tb=short -v

test-sam3-video:
	$(ML_COMPOSE) exec sam3-video-backend python -m pytest tests/ --tb=short -v

test-sam21-image:
	$(ML_COMPOSE) exec sam21-image-backend python -m pytest tests/ --tb=short -v

test-sam21-video:
	$(ML_COMPOSE) exec sam21-video-backend python -m pytest tests/ --tb=short -v

# ─── Supabase Admin Overlay (Studio + Meta API) ─────────────
supabase-up:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f .env.supabase || (echo "Missing .env.supabase. Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_COMPOSE) up -d supabase-meta supabase-studio

supabase-down:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f .env.supabase || (echo "Missing .env.supabase. Run: cp .env.supabase.example .env.supabase" && exit 1)
	-$(SUPABASE_COMPOSE) stop supabase-studio supabase-meta
	-$(SUPABASE_COMPOSE) rm -f supabase-studio supabase-meta

supabase-logs:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f .env.supabase || (echo "Missing .env.supabase. Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_COMPOSE) logs -f --tail=100 supabase-studio supabase-meta

# ─── Supabase S3 Storage Overlay (profile: supabase-s3) ─────
supabase-s3-up:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f .env.supabase || (echo "Missing .env.supabase. Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_S3_COMPOSE) up -d supabase-imgproxy supabase-storage

supabase-s3-down:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f .env.supabase || (echo "Missing .env.supabase. Run: cp .env.supabase.example .env.supabase" && exit 1)
	-$(SUPABASE_S3_COMPOSE) stop supabase-storage supabase-imgproxy
	-$(SUPABASE_S3_COMPOSE) rm -f supabase-storage supabase-imgproxy

supabase-s3-logs:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f .env.supabase || (echo "Missing .env.supabase. Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_S3_COMPOSE) logs -f --tail=100 supabase-storage supabase-imgproxy

# ─── Developer Tools ─────────────────────────────────────────
tools-up:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f .env.tools || (echo "Missing .env.tools. Run: cp .env.tools.example .env.tools" && exit 1)
	$(TOOLS_COMPOSE) up -d redisinsight

tools-down:
	-$(TOOLS_COMPOSE_BASE) stop redisinsight
	-$(TOOLS_COMPOSE_BASE) rm -f redisinsight

tools-logs:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f .env.tools || (echo "Missing .env.tools. Run: cp .env.tools.example .env.tools" && exit 1)
	$(TOOLS_COMPOSE) logs -f --tail=100 redisinsight

# ─── Initialisation ──────────────────────────────────────────
init-minio:
	docker compose run --rm minio-init

create-admin:
	@bash scripts/create-admin.sh

reset-password:
	@bash scripts/reset-password.sh

# ─── Health check ────────────────────────────────────────────
health:
	@bash scripts/healthcheck.sh

# ─── Git ─────────────────────────────────────────────────────
push:
	git add -A
	@read -p "Commit message: " msg; git commit -m "$$msg"
	git push origin main
