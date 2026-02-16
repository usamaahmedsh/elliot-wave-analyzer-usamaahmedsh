#!/bin/bash
# Elliott Wave Analyzer - Quick Start
# 
# Minimal one-command pipeline for quick testing
# Usage: ./quickstart.sh

set -e

echo "🚀 Elliott Wave Analyzer - Quick Start"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Setting up environment (first-time setup)..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo "✅ Environment ready"
else
    echo "✅ Using existing environment"
    source .venv/bin/activate
fi

echo ""
echo "🔍 Running pattern detection on AAPL..."
export PYTHONPATH=.
python scripts/pipeline_run.py --symbols AAPL --config configs.yaml --output output/quickstart_results.json

echo ""
echo "📊 Evaluating results..."
python utils/evaluation/evaluate_patterns.py --mode rules --input output/quickstart_results.json

echo ""
echo "✅ Complete! Results saved to: output/quickstart_results.json"
echo ""
echo "Next steps:"
echo "  - Run full pipeline: bin/run_pipeline.sh AAPL,MSFT,GOOG"
echo "  - View results: cat output/quickstart_results.json | jq '.'"
echo "  - High-performance: bin/run_pipeline.sh --config configs_high_perf.yaml"
