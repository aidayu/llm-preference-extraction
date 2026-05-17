#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = llm-preference-extraction
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
UV ?= uv
DATASET ?= data/ground_truth/test.json
SAVE_DIR ?= data/results/raw/experiments
RESULTS_PATH ?=
EVAL_OUT_DIR ?=
CSV_PATH ?=
PLOT_COMPARE ?=
PLOT_TYPE ?= both

SHELL := /bin/bash

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## セットアップ: 仮想環境作成と依存関係インストール
.PHONY: setup
setup:
	$(UV) venv --python $(PYTHON_VERSION)
	$(UV) sync
	@echo ">>> セットアップ完了。以下でアクティベート:"
	@echo ">>> source .venv/bin/activate"

## 依存関係インストール
.PHONY: install
install:
	$(UV) sync

## 環境チェック
.PHONY: verify
verify:
	@command -v $(UV) >/dev/null 2>&1 || (echo "uv が見つかりません。PATH か UV 変数を確認してください。"; exit 1)
	@$(PYTHON_INTERPRETER) -c "import sys; print(sys.version); sys.exit(0 if sys.version_info >= (3, 11) else 1)" || (echo "Python 3.11+ が必要です。"; exit 1)
	@test -f .env || (echo ".env がありません。cp .env.example .env を実行してください。"; exit 1)
	@echo ">>> OK"

## テスト実行
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests -v

## リント (ruff)
.PHONY: lint
lint:
	ruff check llm_preference_extraction experiments app

## コード整形
.PHONY: format
format:
	ruff check --fix llm_preference_extraction experiments app
	ruff format llm_preference_extraction experiments app

## Streamlit アプリ起動
.PHONY: app
app:
	streamlit run app/main.py

## 抽出実験実行 (例: make extract MODEL=gpt-4o)
MODEL ?= gpt-4o-mini
.PHONY: extract
extract:
	$(PYTHON_INTERPRETER) experiments/run_extraction.py --model $(MODEL) --dataset $(DATASET) --save-dir $(SAVE_DIR)

## 評価実行
.PHONY: evaluate
evaluate:
	@if [[ -z "$(RESULTS_PATH)" ]]; then echo "RESULTS_PATH を指定してください。"; exit 1; fi
	$(PYTHON_INTERPRETER) experiments/run_evaluation.py --results $(RESULTS_PATH) $(if $(EVAL_OUT_DIR),--output-dir $(EVAL_OUT_DIR),)

## グラフ描画 (例: make plot RESULTS_PATH=data/results/raw/.../experiment_results_*.json)
.PHONY: plot
plot:
	@if [[ -n "$(CSV_PATH)" ]]; then \
		$(PYTHON_INTERPRETER) experiments/plot_evaluation.py "$(CSV_PATH)" $(if $(PLOT_COMPARE),--compare,) --type $(PLOT_TYPE); \
	elif [[ -n "$(RESULTS_PATH)" ]]; then \
		_ts=$$(python -c "import re; m=re.search(r'(\d{8}_\d{6})', '$(RESULTS_PATH)'); print(m.group(1) if m else '')"); \
		_model=$$(python -c "from pathlib import Path; p=Path('$(RESULTS_PATH)'); print(p.parent.name)"); \
		_dir="data/results/raw/evaluations/$$_model/$$_ts"; \
		_csv=$$(ls "$$_dir"/evaluation_*.csv 2>/dev/null | sort | tail -1); \
		if [[ -z "$$_csv" ]]; then echo "Error: 評価CSVが見つかりません: $$_dir"; exit 1; fi; \
		echo "評価CSVを使用: $$_csv"; \
		$(PYTHON_INTERPRETER) experiments/plot_evaluation.py "$$_csv" --type $(PLOT_TYPE); \
	else \
		$(PYTHON_INTERPRETER) experiments/plot_evaluation.py; \
	fi

## キャッシュ削除
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('利用可能なコマンド:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
