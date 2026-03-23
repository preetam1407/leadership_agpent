#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

mkdir -p outputs/sample_answers outputs/sample_plots
find outputs/sample_answers -maxdepth 1 -type f -delete
find outputs/sample_plots -maxdepth 1 -type f -delete

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

"$PYTHON_BIN" main.py ingest --config config.yaml
"$PYTHON_BIN" main.py build-index --config config.yaml
"$PYTHON_BIN" main.py ask --config config.yaml --question "What is our current revenue trend?" --report_path outputs/sample_answers/revenue_trend.json
"$PYTHON_BIN" main.py ask --config config.yaml --question "Which departments are underperforming?" --report_path outputs/sample_answers/underperforming_departments.json
"$PYTHON_BIN" main.py ask --config config.yaml --question "What were the key risks highlighted in the last quarter?" --report_path outputs/sample_answers/key_risks_last_quarter.json
"$PYTHON_BIN" main.py ask --config config.yaml --question "How did Digital Media and Digital Experience perform in Q2 FY2025?" --report_path outputs/sample_answers/segment_comparison.json
"$PYTHON_BIN" main.py ask --config config.yaml --question "What does Adobe emphasize about AI, Firefly and Acrobat AI Assistant?" --report_path outputs/sample_answers/ai_strategy.json
"$PYTHON_BIN" main.py eval --config config.yaml --results_path data/eval/results.json
"$PYTHON_BIN" main.py plot --config config.yaml --results_path data/eval/results.json --output_dir outputs/sample_plots

printf '\nDemo complete. Answers are in outputs/sample_answers and plots are in outputs/sample_plots.\n'
