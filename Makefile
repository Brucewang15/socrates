MODEL_PORT    ?= 8080
BACKEND_PORT  ?= 8000
MODEL_URL     ?= http://localhost:$(MODEL_PORT)

# frontend -> backend -> model
dev:
	@trap "kill 0" INT TERM EXIT; \
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

install:
	uv sync --all-extras
	npm --prefix frontend install

# Both GPU-host images, from the same docker-compose.yaml the instance runs.
# Building works anywhere; running needs an NVIDIA host.
images:
	docker compose build

push:
	./push-to-ecr.sh

.PHONY: dev model backend frontend install images push
