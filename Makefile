SHELL := /bin/bash

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
ACC ?= .venv/bin/accelerate

DATASETS_TXT ?= datasets.txt
ART ?= ml/data/artifacts
META_JSONL ?= $(ART)/manifests/datasets_meta.jsonl
BINS_JSON ?= $(ART)/manifests/bins.json
COVERAGE_OUT ?= $(ART)/manifests/coverage.json
COVERAGE_BASE_MODEL ?=
COVERAGE_TRACK ?=

LANG ?=
TRACK ?= cc0_pd
LIMIT ?= 20

BASE_MODEL ?= mistralai/Ministral-3-14B-Base-2512
RUN_TAG ?= $(shell date -u +%Y%m%dT%H%M%SZ)

NORM_DIR = $(ART)/normalized/lang=$(LANG)/track=$(TRACK)
RUN_DIR = $(ART)/runs/base=$(subst /,_,$(BASE_MODEL))/lang=$(LANG)/track=$(TRACK)/$(RUN_TAG)

.PHONY: venv deps env-check mdc-meta mdc-bins mdc-download data stats coverage train eval publish list-bins clean

venv:
	python -m venv .venv

deps: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

env-check:
	@test -n "$$MDC_API_KEY" || (echo "MDC_API_KEY is not set"; exit 1)

mdc-meta: deps env-check
	$(PY) -m ml.data.etl meta --datasets-txt $(DATASETS_TXT) --out $(META_JSONL)

mdc-bins: deps
	$(PY) -m ml.data.etl bins --meta $(META_JSONL) --out $(BINS_JSON)

mdc-download: deps env-check
	$(PY) -m ml.data.etl download --bins $(BINS_JSON) --limit $(LIMIT) $(if $(LANG),--lang $(LANG),) $(if $(TRACK),--track $(TRACK),)

data: deps
	@test -n "$(LANG)" || (echo "LANG is required (e.g., LANG=fi)"; exit 1)
	$(PY) -m ml.data.etl build --bins $(BINS_JSON) --lang $(LANG) --track $(TRACK)

stats: deps
	@test -n "$(LANG)" || (echo "LANG is required"; exit 1)
	$(PY) -m ml.data.etl stats --lang $(LANG) --track $(TRACK)

coverage: deps
	$(PY) -m ml.data.etl coverage --out $(COVERAGE_OUT) $(if $(wildcard $(BINS_JSON)),--bins $(BINS_JSON),) $(if $(COVERAGE_BASE_MODEL),--base-model $(COVERAGE_BASE_MODEL),) $(if $(COVERAGE_TRACK),--track $(COVERAGE_TRACK),)

train: deps
	@test -n "$(LANG)" || (echo "LANG is required"; exit 1)
	@test -f "$(NORM_DIR)/train.jsonl.zst" || (echo "Missing train shards. Run: make data LANG=$(LANG) TRACK=$(TRACK)"; exit 1)
	$(ACC) launch -m ml.models.train \
		--base-model $(BASE_MODEL) \
		--train-zst $(NORM_DIR)/train.jsonl.zst \
		--valid-zst $(NORM_DIR)/valid.jsonl.zst \
		--out $(RUN_DIR)

eval: deps
	@test -n "$(LANG)" || (echo "LANG is required"; exit 1)
	$(PY) -m ml.models.eval --base-model $(BASE_MODEL) --adapter $(RUN_DIR) --valid-zst $(NORM_DIR)/valid.jsonl.zst --out $(RUN_DIR)/eval.json

publish: deps
	@test -n "$(HF_REPO)" || (echo "HF_REPO is required (e.g., HF_REPO=myorg/ministral14b-lora-fi-cc0)"; exit 1)
	$(PY) -m ml.models.publish --repo $(HF_REPO) --adapter $(RUN_DIR)

list-bins: deps
	$(PY) -m ml.data.etl list --bins $(BINS_JSON)

clean:
	rm -rf $(ART)/runs
