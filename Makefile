PYTHON := .venv/bin/python
UV := uv

.PHONY: setup test lint link-audit-sample coupling-comparison clean

setup:
	$(UV) sync --python 3.12 --all-extras

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

link-audit-sample:
	$(PYTHON) -m noether_link_audit --config configs/link-audit-sample.yaml --output results/link-audit-sample-v1 --work results/runs/link-audit-sample-v1

coupling-comparison:
	$(PYTHON) scripts/compare_fiber_coupling.py > results/fiber-coupling-v1/comparison.json

clean:
	find . -type d -name __pycache__ -prune -exec rm -r {} +
