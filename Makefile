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
		check-core-env check-supabase-standalone-env check-supabase-sample-env check-tools-env check-ls-supavisor-user-format check-postgrest-schema-exposure \
		tools-up tools-down tools-logs \
        init-minio health create-admin reset-password \
        push \
        batch-annotate batch-server

# ─── Core stack ─────────────────────────────────────────────
CORE_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME)

up: check-ls-supavisor-user-format supabase-up
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
SUPABASE_STANDALONE_ENV := $(if $(strip $(SUPABASE_STANDALONE_ENV)),$(SUPABASE_STANDALONE_ENV),.env.supabase)
SUPABASE_STANDALONE_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME) --env-file .env --env-file $(SUPABASE_STANDALONE_ENV) -f docker-compose.supabase.yml
SUPABASE_SAMPLE_ENV ?= .env.supabase.sample
SUPABASE_SAMPLE_ENV := $(if $(strip $(SUPABASE_SAMPLE_ENV)),$(SUPABASE_SAMPLE_ENV),.env.supabase.sample)
SUPABASE_SAMPLE_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME) --env-file .env --env-file $(SUPABASE_SAMPLE_ENV) -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.supabase.sample.yml
TOOLS_COMPOSE_BASE = docker compose --project-name $(STACK_PROJECT_NAME) -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.tools.yml
TOOLS_COMPOSE = docker compose --project-name $(STACK_PROJECT_NAME) --env-file .env --env-file .env.tools -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.tools.yml

# ─── Env checks (cross-platform, GNU make-native) ──────────
check-core-env:
	$(if $(wildcard .env),,$(error Missing .env. Run: cp .env.example .env))

check-supabase-standalone-env: check-core-env
	$(if $(wildcard $(SUPABASE_STANDALONE_ENV)),,$(error Missing $(SUPABASE_STANDALONE_ENV). Run: cp .env.supabase.example .env.supabase))

check-supabase-sample-env: check-core-env
	$(if $(wildcard $(SUPABASE_SAMPLE_ENV)),,$(error Missing $(SUPABASE_SAMPLE_ENV). Run: cp .env.supabase.sample.template .env.supabase.sample))

check-tools-env: check-core-env
	$(if $(wildcard .env.tools),,$(error Missing .env.tools. Run: cp .env.tools.example .env.tools))

check-ls-supavisor-user-format: check-supabase-standalone-env
	@bash scripts/validate-supavisor-user.sh "$(SUPABASE_STANDALONE_ENV)"

check-postgrest-schema-exposure: check-supabase-standalone-env
	@bash scripts/validate-postgrest-security.sh "$(SUPABASE_STANDALONE_ENV)"

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
	$(ML_COMPOSE) build sam3-video-craft

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
supabase-up: check-supabase-standalone-env check-postgrest-schema-exposure
	$(SUPABASE_STANDALONE_COMPOSE) up -d

supabase-down: check-supabase-standalone-env
	$(SUPABASE_STANDALONE_COMPOSE) down

supabase-logs: check-supabase-standalone-env
	$(SUPABASE_STANDALONE_COMPOSE) logs -f --tail=100 studio meta db

# Explicit aliases for standalone mode.
supabase-standalone-up: supabase-up

supabase-standalone-down: supabase-down

supabase-standalone-logs: supabase-logs

# Supabase minimal example mode (studio + meta only).
supabase-sample-up: check-supabase-sample-env
	$(SUPABASE_SAMPLE_COMPOSE) up -d supabase-studio supabase-meta

supabase-sample-down:
	-$(SUPABASE_SAMPLE_COMPOSE) stop supabase-studio supabase-meta
	-$(SUPABASE_SAMPLE_COMPOSE) rm -f supabase-studio supabase-meta

supabase-sample-logs: check-supabase-sample-env
	$(SUPABASE_SAMPLE_COMPOSE) logs -f --tail=100 supabase-studio supabase-meta

# ─── Developer Tools ─────────────────────────────────────────
tools-up: check-tools-env
	$(TOOLS_COMPOSE) up -d redisinsight

tools-down:
	-$(TOOLS_COMPOSE_BASE) stop redisinsight
	-$(TOOLS_COMPOSE_BASE) rm -f redisinsight

tools-logs: check-tools-env
	$(TOOLS_COMPOSE) logs -f --tail=100 redisinsight

# ─── Initialisation ──────────────────────────────────────────
init-minio:
	@bash -lc 'COMPOSE_PROJECT_NAME="$(STACK_PROJECT_NAME)" docker compose run --rm minio-init'

create-admin:
	@bash -lc 'COMPOSE_PROJECT_NAME="$(STACK_PROJECT_NAME)" bash scripts/create-admin.sh'

reset-password:
	@bash -lc 'COMPOSE_PROJECT_NAME="$(STACK_PROJECT_NAME)" bash scripts/reset-password.sh'

# ─── Health check ────────────────────────────────────────────
health:
	@bash -lc 'COMPOSE_PROJECT_NAME="$(STACK_PROJECT_NAME)" bash scripts/healthcheck.sh'

# ─── Batch Annotation ────────────────────────────────────────
batch-annotate: check-core-env
	@[ -n "$(PROJECT_ID)" ] || (echo "Usage: make batch-annotate PROJECT_ID=<id> [BACKEND=sam3|sam21]"; exit 1)
	LABEL_STUDIO_API_KEY=$$(grep '^LABEL_STUDIO_API_KEY=' .env | cut -d= -f2-) \
	  python scripts/batch_annotate.py \
	    --project-id $(PROJECT_ID) \
	    --backend $(or $(BACKEND),sam3) \
	    --ls-url $$(grep '^LABEL_STUDIO_EXTERNAL_URL=' .env | cut -d= -f2- || echo http://localhost:8080) \
	    $(if $(CONFIDENCE),--confidence $(CONFIDENCE),) \
	    $(if $(MAX_TASKS),--max-tasks $(MAX_TASKS),) \
	    $(if $(DRY_RUN),--dry-run,)

batch-server:
	$(CORE_COMPOSE) up -d --build batch-server

# ─── Git ─────────────────────────────────────────────────────
push:
	git add -A
	@read -p "Commit message: " msg; git commit -m "$$msg"
	git push origin main
