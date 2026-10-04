PY ?= .venv/bin/python

.PHONY: all quick test venv clean-results

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

test:
	$(PY) -m pytest -q -p no:warnings

all:
	scripts/run_all.sh

quick:
	QUICK=1 scripts/run_all.sh

clean-results:
	rm -rf results/e1_open_loop results/e2_data_efficiency results/e3_closed_loop results/e4_timing results/e5_robustness results/e6_ablations results/models results/logs results/MANIFEST.md
