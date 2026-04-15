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
		supabase-sample-up supabase-sample-down supabase-sample-logs \
	tools-up tools-down tools-logs \
        init-minio health create-admin reset-password \
        push

# ─── Core stack ─────────────────────────────────────────────
CORE_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME)

up:
	$(CORE_COMPOSE) up -d

down:
	$(CORE_COMPOSE) down

restart:
	$(CORE_COMPOSE) restart

logs:
	$(CORE_COMPOSE) logs -f --tail=100

ps:
	$(CORE_COMPOSE) ps

# ─── ML Backends (SAM3 + SAM2.1 image + video) ──────────────
# override.yml must be included explicitly when using -f flags
# (Docker Compose only auto-loads override.yml when no -f is specified)
ML_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME) -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml
STACK_PROJECT_NAME ?= label-anything-sam
SUPABASE_STANDALONE_ENV ?= .env.supabase
SUPABASE_STANDALONE_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME) --env-file .env --env-file $(SUPABASE_STANDALONE_ENV) -f docker-compose.supabase.yml
SUPABASE_SAMPLE_ENV ?= .env.supabase.sample
SUPABASE_SAMPLE_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME) --env-file .env --env-file $(SUPABASE_SAMPLE_ENV) -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.supabase.sample.yml
TOOLS_COMPOSE_BASE = docker compose --project-name $(STACK_PROJECT_NAME) -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.tools.yml
TOOLS_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME) --env-file .env --env-file .env.tools -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.tools.yml

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
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f $(SUPABASE_STANDALONE_ENV) || (echo "Missing $(SUPABASE_STANDALONE_ENV). Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_STANDALONE_COMPOSE) up -d

supabase-down:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f $(SUPABASE_STANDALONE_ENV) || (echo "Missing $(SUPABASE_STANDALONE_ENV). Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_STANDALONE_COMPOSE) down

supabase-logs:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f $(SUPABASE_STANDALONE_ENV) || (echo "Missing $(SUPABASE_STANDALONE_ENV). Run: cp .env.supabase.example .env.supabase" && exit 1)
	$(SUPABASE_STANDALONE_COMPOSE) logs -f --tail=100 studio meta db

# Explicit aliases for standalone mode.
supabase-standalone-up: supabase-up

supabase-standalone-down: supabase-down

supabase-standalone-logs: supabase-logs

# Supabase minimal example mode (studio + meta only).
supabase-sample-up:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f $(SUPABASE_SAMPLE_ENV) || (echo "Missing $(SUPABASE_SAMPLE_ENV). Run: cp .env.supabase.sample.template .env.supabase.sample" && exit 1)
	$(SUPABASE_SAMPLE_COMPOSE) up -d supabase-studio supabase-meta

supabase-sample-down:
	-$(SUPABASE_SAMPLE_COMPOSE) stop supabase-studio supabase-meta
	-$(SUPABASE_SAMPLE_COMPOSE) rm -f supabase-studio supabase-meta

supabase-sample-logs:
	@test -f .env || (echo "Missing .env. Run: cp .env.example .env" && exit 1)
	@test -f $(SUPABASE_SAMPLE_ENV) || (echo "Missing $(SUPABASE_SAMPLE_ENV). Run: cp .env.supabase.sample.template .env.supabase.sample" && exit 1)
	$(SUPABASE_SAMPLE_COMPOSE) logs -f --tail=100 supabase-studio supabase-meta

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
	COMPOSE_PROJECT_NAME=$(STACK_PROJECT_NAME) docker compose run --rm minio-init

create-admin:
	@COMPOSE_PROJECT_NAME=$(STACK_PROJECT_NAME) bash scripts/create-admin.sh

reset-password:
	@COMPOSE_PROJECT_NAME=$(STACK_PROJECT_NAME) bash scripts/reset-password.sh

# ─── Health check ────────────────────────────────────────────
health:
	@COMPOSE_PROJECT_NAME=$(STACK_PROJECT_NAME) bash scripts/healthcheck.sh

# ─── Git ─────────────────────────────────────────────────────
push:
	git add -A
	@read -p "Commit message: " msg; git commit -m "$$msg"
	git push origin main
