PYTHON ?= python
RUFF ?= ruff

# Limit formatting to the maintained value-method additions and adapters.
VALUE_CODE = src/model/value_model.py src/trainer/value_trainer.py \
	src/trainer/value_targets.py src/trainer/exact_ppo.py \
	src/trainer/__init__.py src/agent/__init__.py src/utils/agents.py \
	src/utils/environment.py src/utils/seed.py src/utils/exploration.py \
	src/utils/memory.py src/eval/value.py src/eval/value_results.py \
	scripts/train_value.py scripts/evaluate.py scripts/verify_value_results.py \
	tests/test_value_learning.py tests/test_value_training.py tests/test_value_evaluation.py \
	tests/test_evaluation_inputs.py tests/test_value_results.py
SHARED_CODE = src/trainer/policy_trainer.py src/eval/protocol.py
REPORT_CODE = scripts/html_report.py scripts/export_checkpoints.py

.PHONY: lint format test check
lint:
	$(RUFF) check $(VALUE_CODE) $(SHARED_CODE)
	$(RUFF) check --select E4,E7,E9,F $(REPORT_CODE)
	$(RUFF) format --check $(VALUE_CODE)

format:
	$(RUFF) check --select I --fix $(VALUE_CODE) $(SHARED_CODE)
	$(RUFF) format $(VALUE_CODE)

test:
	$(PYTHON) -m pytest

check: lint test
