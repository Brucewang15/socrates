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

images:
	docker build -f model/Dockerfile    -t socrates-model    .
	docker build -f backend/Dockerfile  -t socrates-backend  .
	docker build -f frontend/Dockerfile -t socrates-frontend .

push:
	./infra/push-to-ecr.sh

.PHONY: dev model backend frontend install images push
