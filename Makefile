ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

.PHONY: install test lint validate run eval

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src/ app/ eval/ tests/
	$(PY) -m ruff format --check src/ app/ eval/ tests/

validate: lint test

run:
	$(PY) -m streamlit run app/streamlit_app.py

eval:
	$(PY) eval/run_eval.py
