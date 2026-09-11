dev:
	@trap "kill 0" INT TERM EXIT; \
	uv run uvicorn backend.server:app --reload --port 8000 & \
	npm --prefix frontend run dev & \
	wait

backend:
	uv run uvicorn backend.server:app --reload --port 8000

frontend:
	npm --prefix frontend run dev

install:
	uv sync
	npm --prefix frontend install

.PHONY: dev backend frontend install
