#!/bin/bash
set -e

# ============================================================
# Elliott Wave Analyzer - Pipeline Runner (CPU / Numba only)
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output"
CONFIG_FILE="$PROJECT_ROOT/configs.yaml"
DATA_DIR="$PROJECT_ROOT/data/hf_dataset_complete"
INTERVALS="1d,4h,1h,1wk"

cleanup() {
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}============================================${NC}"
        echo -e "${GREEN}  Pipeline completed successfully!${NC}"
        echo -e "${GREEN}============================================${NC}"
    else
        echo -e "${RED}============================================${NC}"
        echo -e "${RED}  Pipeline exited with error (code: $EXIT_CODE)${NC}"
        echo -e "${RED}============================================${NC}"
    fi
}
trap cleanup EXIT

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Elliott Wave Analyzer Pipeline${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "${BLUE}Project : $PROJECT_ROOT${NC}"
echo -e "${BLUE}Date    : $(date)${NC}"
echo -e "${BLUE}Workers : ${SLURM_NTASKS:-$(nproc)}${NC}"
echo ""

# ============================================================
# Step 1: Load HPC Python module (no CUDA needed)
# ============================================================
echo -e "${YELLOW}[1/5] Loading HPC modules...${NC}"
if command -v module &> /dev/null; then
    module load python3/3.12.4 2>/dev/null || \
    module load python3/3.11   2>/dev/null || \
    module load python/3.11    2>/dev/null || \
    echo -e "${YELLOW}  No python module loaded (using system python)${NC}"
    echo -e "${GREEN}  Done${NC}"
else
    echo -e "${YELLOW}  Local environment (no module system)${NC}"
fi
echo ""

# ============================================================
# Step 2: Activate existing .venv (created by user)
# ============================================================
echo -e "${YELLOW}[2/5] Activating virtual environment...${NC}"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo -e "${RED}  No .venv found. Create one first: python3 -m venv .venv && pip install -r requirements.txt${NC}"
    exit 1
fi
echo -e "${GREEN}  Python: $(python --version)${NC}"
echo ""

# ============================================================
# Step 3: Prepare output directory
# ============================================================
echo -e "${YELLOW}[3/5] Preparing output directory...${NC}"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/images"
mkdir -p "$PROJECT_ROOT/logs"
echo -e "${GREEN}  Output dir ready: $OUTPUT_DIR${NC}"
echo ""

# ============================================================
# Step 4: Run Elliott Wave pipeline
# ============================================================
echo -e "${YELLOW}[4/5] Running pattern detection...${NC}"
echo -e "${BLUE}  Data dir  : $DATA_DIR${NC}"
echo -e "${BLUE}  Intervals : $INTERVALS${NC}"
echo -e "${BLUE}  Config    : $CONFIG_FILE${NC}"
echo ""

START_TIME=$(date +%s)

python scripts/pipeline_run.py \
    --data-dir       "$DATA_DIR" \
    --config         "$CONFIG_FILE" \
    --output         "$OUTPUT_DIR" \
    --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
    --intervals      "$INTERVALS" \
    --resume \
    --verbose

END_TIME=$(date +%s)
RUNTIME=$(( END_TIME - START_TIME ))
echo ""
echo -e "${GREEN}  Pattern detection complete  (${RUNTIME}s / $(( RUNTIME/60 ))m$(( RUNTIME%60 ))s)${NC}"
echo ""

# ============================================================
# Step 5: Run evaluation scripts
# ============================================================
echo -e "${YELLOW}[5/5] Running evaluation...${NC}"

RESULTS_FILE="$OUTPUT_DIR/results.json"
if [ ! -f "$RESULTS_FILE" ]; then
    echo -e "${YELLOW}  results.json not found, skipping evaluation${NC}"
    exit 0
fi

PATTERN_COUNT=$(python -c "import json; d=json.load(open('$RESULTS_FILE')); print(len(d.get('patterns', d)))")
echo -e "${GREEN}  Patterns found: $PATTERN_COUNT${NC}"

for SCRIPT in tools/evaluate_data_distribution.py tools/validate_patterns_v2.py tools/evaluate_rule_validity.py; do
    if [ -f "$PROJECT_ROOT/$SCRIPT" ]; then
        OUTFILE="$OUTPUT_DIR/$(basename ${SCRIPT%.py}).json"
        python "$PROJECT_ROOT/$SCRIPT" --input "$RESULTS_FILE" --output "$OUTFILE" && \
            echo -e "${GREEN}  $SCRIPT -> $OUTFILE${NC}" || \
            echo -e "${YELLOW}  $SCRIPT failed (non-fatal)${NC}"
    fi
done
echo ""

# Final summary
echo -e "${GREEN}============================================${NC}"
echo -e "${BLUE}Output files:${NC}"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (none)"
echo -e "${BLUE}Total runtime: $(( RUNTIME/60 ))m $(( RUNTIME%60 ))s${NC}"
