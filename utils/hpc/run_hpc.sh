#!/bin/bash
# Elliott Wave Analyzer - HPC Quick Start
# 
# Optimized for HPC environments with module system
# Usage: ./run_hpc.sh [SYMBOLS]

set -e

echo "🚀 Elliott Wave Analyzer - HPC Mode"
echo ""

# Default symbols
SYMBOLS="${1:-AAPL,MSFT,GOOG}"

# Load modules
echo "📦 Loading HPC modules..."
module load python3/3.10.12 || module load python3/3.10.5 || module load python3
module load cuda/12.5 || module load cuda/12.2 || module load cuda

echo ""
echo "Loaded modules:"
module list

echo ""
echo "🔧 Installing dependencies to user directory..."
pip3 install --user -r requirements.txt --quiet || pip3 install --user -r requirements.txt

echo ""
echo "💾 Creating output directory..."
OUTPUT_DIR="output/hpc_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "🔍 Running pattern detection on: $SYMBOLS"
echo "   Using HF dataset: usamaahmedsh/financial-markets-dataset-15y-train"
echo "   Output: $OUTPUT_DIR/results.json"
echo ""

# Run pipeline
export PYTHONPATH=.
python3 scripts/pipeline_run_enhanced.py \
    --symbols "$SYMBOLS" \
    --config configs.yaml \
    --output "$OUTPUT_DIR/results.json" \
    --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
    --hf-dataset "usamaahmedsh/financial-markets-dataset-15y-train" \
    --verbose

echo ""
echo "✅ Complete! Results saved to: $OUTPUT_DIR/results.json"
echo ""
echo "Next steps:"
echo "  - View results: cat $OUTPUT_DIR/results.json | jq '.'"
echo "  - Run evaluation: python3 utils/evaluation/evaluate_patterns.py --mode all --input $OUTPUT_DIR/results.json"
echo "  - Resume if interrupted: bin/run_pipeline.sh --hpc --resume $OUTPUT_DIR"
