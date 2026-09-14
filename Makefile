MODEL_PORT    ?= 8080
BACKEND_PORT  ?= 8000
MODEL_URL     ?= http://localhost:$(MODEL_PORT)
COMPOSE_DEV   := docker compose -f docker-compose.yaml -f docker-compose.dev.yml

# frontend -> backend -> model, plus prometheus/grafana scraping the two servers.
# dcgm-exporter is skipped: it needs an NVIDIA host.
dev: monitoring
	@echo "prometheus http://localhost:9090   grafana http://localhost:3001/d/socrates"
	@trap "kill 0; $(COMPOSE_DEV) stop prometheus grafana >/dev/null 2>&1" INT TERM EXIT; \
	uv run uvicorn model.server:app --port $(MODEL_PORT) & \
	MODEL_URL=$(MODEL_URL) uv run uvicorn backend.server:app --reload --port $(BACKEND_PORT) & \
	npm --prefix frontend run dev & \
	wait

# No --reload on the model tier: every reload re-reads 8 GB of weights.
model:
	uv run uvicorn model.server:app --port $(MODEL_PORT)

backend:
	MODEL_URL=$(MODEL_URL) uv run uvicorn backend.server:app --reload --port $(BACKEND_PORT)

frontend:
	npm --prefix frontend run dev

monitoring:
	@$(COMPOSE_DEV) up -d prometheus grafana

monitoring-down:
	@$(COMPOSE_DEV) down prometheus grafana

install:
	uv sync --all-extras
	npm --prefix frontend install

# Both GPU-host images, from the same docker-compose.yaml the instance runs.
# Building works anywhere; running needs an NVIDIA host.
images:
	docker compose build

push:
	./push-to-ecr.sh

.PHONY: dev model backend frontend monitoring monitoring-down install images push
