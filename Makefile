SHELL := /bin/bash

PY  ?= .venv/bin/python
PIP ?= .venv/bin/pip

# Prevent system LANG (locale) from leaking into our L variable
unexport LANG

# Legacy ETL paths (used by init to bootstrap from MDC metadata)
ART            ?= ml/data/artifacts
DATASETS_TXT   ?= datasets.txt
META_JSONL     ?= $(ART)/manifests/datasets_meta.jsonl
BINS_JSON      ?= $(ART)/manifests/bins.json

# Per-language parameters — use L= for brevity (e.g., make status L=fi)
L ?=
LIMIT ?= 100
MAX_GB ?=
BASE_MODEL ?=
RUN_TAG ?=
HF_REPO ?=
TPU ?=

# TPU flag passthrough
TPU_FLAG = $(if $(filter 1 true yes,$(TPU)),--tpu,)

.PHONY: venv deps deps-tpu env-check \
        mdc-meta mdc-bins \
        init init-all download build train eval status publish migrate \
        run viz clean

# ── Setup ─────────────────────────────────────────────────────────
venv:
	python -m venv .venv

deps: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

deps-tpu: deps
	$(PIP) install -r requirements-tpu.txt

env-check:
	@test -n "$$MDC_API_KEY" || (echo "MDC_API_KEY is not set"; exit 1)

# ── MDC metadata (run once to bootstrap) ──────────────────────────
mdc-meta: deps env-check
	$(PY) -m ml.data.etl meta --datasets-txt $(DATASETS_TXT) --out $(META_JSONL)

mdc-bins: deps
	$(PY) -m ml.data.etl bins --meta $(META_JSONL) --out $(BINS_JSON)

# ── Per-language pipeline ─────────────────────────────────────────
init: deps
	@test -n "$(L)" || (echo "L is required. Use init-all for all languages."; exit 1)
	$(PY) -m polyglot init --lang $(L) --bins $(BINS_JSON)

init-all: deps
	$(PY) -m polyglot init --all --bins $(BINS_JSON)

download: deps env-check
	@test -n "$(L)" || (echo "L is required (e.g., L=fi)"; exit 1)
	$(PY) -m polyglot download --lang $(L) $(if $(LIMIT),--limit $(LIMIT),) $(if $(MAX_GB),--max-gb $(MAX_GB),)

build: deps
	@test -n "$(L)" || (echo "L is required"; exit 1)
	$(PY) -m polyglot build --lang $(L)

train: deps
	@test -n "$(L)" || (echo "L is required"; exit 1)
	$(PY) -m polyglot train --lang $(L) $(if $(BASE_MODEL),--base-model $(BASE_MODEL),) $(if $(RUN_TAG),--run-tag $(RUN_TAG),) $(TPU_FLAG)

eval: deps
	@test -n "$(L)" || (echo "L is required"; exit 1)
	$(PY) -m polyglot eval --lang $(L) $(if $(BASE_MODEL),--base-model $(BASE_MODEL),) $(if $(RUN_TAG),--run-tag $(RUN_TAG),) $(TPU_FLAG)

status: deps
	$(PY) -m polyglot status $(if $(L),--lang $(L),)

publish: deps
	@test -n "$(L)" || (echo "L is required"; exit 1)
	@test -n "$(HF_REPO)" || (echo "HF_REPO is required"; exit 1)
	$(PY) -m polyglot publish --lang $(L) --hf-repo $(HF_REPO) $(if $(RUN_TAG),--run-tag $(RUN_TAG),)

migrate: deps
	$(PY) -m polyglot migrate

# ── Convenience ───────────────────────────────────────────────────
run: download build train eval
	@echo "Pipeline complete for $(L)"

viz: deps
	$(PY) -m ml.viz.lang_coverage --out lang_coverage.png

clean:
	rm -rf langs/*/runs
