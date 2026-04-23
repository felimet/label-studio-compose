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
        init-minio health create-admin reset-password \
        push \
        batch-annotate batch-server

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

# ─── Batch Annotation ────────────────────────────────────────
batch-annotate:
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
	docker compose up -d --build batch-server

# ─── Git ─────────────────────────────────────────────────────
push:
	git add -A
	@read -p "Commit message: " msg; git commit -m "$$msg"
	git push origin main
