.PHONY: install test lint run stop demo ui compose-up compose-down

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check reg_agents

run:
	. .venv/bin/activate && bash scripts/run_local.sh

stop:
	bash scripts/stop_local.sh

demo:
	. .venv/bin/activate && python scripts/demo_run.py --model FRAUD-XGB-GNN-001

ui:
	. .venv/bin/activate && streamlit run reg_agents/ui/app.py

compose-up:
	docker compose up --build

compose-down:
	docker compose down
