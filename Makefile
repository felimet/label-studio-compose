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
		supabase-standalone-up supabase-standalone-down supabase-standalone-logs \
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
SUPABASE_STANDALONE_ENV ?= .env.supabase
SUPABASE_STANDALONE_COMPOSE = docker compose --env-file $(SUPABASE_STANDALONE_ENV) -f docker-compose.supabase.yml
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

# ─── Supabase Management (default: standalone) ─────────────
supabase-up:
	@test -f $(SUPABASE_STANDALONE_ENV) || (echo "Missing $(SUPABASE_STANDALONE_ENV). Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_STANDALONE_COMPOSE) up -d

supabase-down:
	@test -f $(SUPABASE_STANDALONE_ENV) || (echo "Missing $(SUPABASE_STANDALONE_ENV). Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_STANDALONE_COMPOSE) down

supabase-logs:
	@test -f $(SUPABASE_STANDALONE_ENV) || (echo "Missing $(SUPABASE_STANDALONE_ENV). Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_STANDALONE_COMPOSE) logs -f --tail=100 studio meta db

# Explicit aliases for standalone mode.
supabase-standalone-up: supabase-up

supabase-standalone-down: supabase-down

supabase-standalone-logs: supabase-logs

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
