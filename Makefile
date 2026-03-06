#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = llm-preference-extraction
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## セットアップ: 仮想環境作成と依存関係インストール
.PHONY: setup
setup:
	uv venv --python $(PYTHON_VERSION)
	uv sync
	@echo ">>> セットアップ完了。以下でアクティベート:"
	@echo ">>> source .venv/bin/activate"

## 依存関係インストール
.PHONY: install
install:
	uv sync

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
	$(PYTHON_INTERPRETER) experiments/run_extraction.py --model $(MODEL)

## 評価実行
.PHONY: evaluate
evaluate:
	$(PYTHON_INTERPRETER) experiments/run_evaluation.py

## グラフ描画
.PHONY: plot
plot:
	$(PYTHON_INTERPRETER) experiments/plot_evaluation.py

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
