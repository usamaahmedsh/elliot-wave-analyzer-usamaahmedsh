#!/bin/bash
set -e

# ============================================================
# Elliott Wave Analyzer - Full Pipeline Runner
# ============================================================
# This script runs the complete Elliott Wave detection pipeline:
# 1. Sets up Python virtual environment
# 2. Loads required HPC modules (CUDA, Python)
# 3. Installs dependencies
# 4. Runs the pattern detection pipeline
# 5. Runs evaluation scripts (EDA + validation)
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
OUTPUT_DIR="$PROJECT_ROOT/output"
CONFIG_FILE="$PROJECT_ROOT/configs.yaml"
RESULTS_FILE="$OUTPUT_DIR/results.json"

# Pipeline settings
USE_HF_DATASET=true  # Set to false to use yfinance instead
HF_DATASET="usamaahmedsh/financial-markets-dataset-15y-train"

# ============================================================
# Cleanup function - runs on script exit
# ============================================================
cleanup() {
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}============================================${NC}"
        echo -e "${GREEN}✓ Pipeline completed successfully!${NC}"
        echo -e "${GREEN}============================================${NC}"
    else
        echo -e "${RED}============================================${NC}"
        echo -e "${RED}✗ Pipeline exited with error (code: $EXIT_CODE)${NC}"
        echo -e "${RED}============================================${NC}"
    fi
}

trap cleanup EXIT

# ============================================================
# Print banner
# ============================================================
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Elliott Wave Analyzer Pipeline${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "${BLUE}Project: $PROJECT_ROOT${NC}"
echo -e "${BLUE}Date: $(date)${NC}"
echo ""

# ============================================================
# Step 1: Load HPC Modules (if available)
# ============================================================
echo -e "${YELLOW}[1/6] Loading HPC modules...${NC}"

# Check if module command exists (HPC environment)
if command -v module &> /dev/null; then
    echo -e "${YELLOW}HPC environment detected, loading modules...${NC}"
    
    # Load CUDA
    if module avail cuda 2>&1 | grep -q "cuda/12"; then
        module load cuda/12.2 2>/dev/null || module load cuda/12.1 2>/dev/null || echo -e "${YELLOW}⚠ CUDA 12.x not available${NC}"
        echo -e "${GREEN}✓ CUDA module loaded${NC}"
    else
        echo -e "${YELLOW}⚠ CUDA module not available${NC}"
    fi
    
    # Load Python
    if module avail python3 2>&1 | grep -q "python3/3.12"; then
        module load python3/3.12.4 2>/dev/null || module load python3/3.11 2>/dev/null || echo -e "${YELLOW}⚠ Python 3.12 not available${NC}"
        echo -e "${GREEN}✓ Python module loaded${NC}"
    elif module avail python 2>&1 | grep -q "python/3.11"; then
        module load python/3.11 2>/dev/null || echo -e "${YELLOW}⚠ Python 3.11 not available${NC}"
        echo -e "${GREEN}✓ Python module loaded${NC}"
    fi
    
    # Show loaded modules
    echo -e "${BLUE}Loaded modules:${NC}"
    module list 2>&1 | grep -v "^$" | head -10
else
    echo -e "${YELLOW}Local environment (no module system)${NC}"
fi
echo ""

# ============================================================
# Step 2: Setup Python Virtual Environment
# ============================================================
echo -e "${YELLOW}[2/6] Setting up Python environment...${NC}"

cd "$PROJECT_ROOT"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    
    # Try different Python versions
    if command -v python3.12 &> /dev/null; then
        echo -e "${YELLOW}Using Python 3.12...${NC}"
        python3.12 -m venv "$VENV_DIR"
    elif command -v python3.11 &> /dev/null; then
        echo -e "${YELLOW}Using Python 3.11...${NC}"
        python3.11 -m venv "$VENV_DIR"
    elif command -v python3 &> /dev/null; then
        echo -e "${YELLOW}Using python3...${NC}"
        python3 -m venv "$VENV_DIR"
    else
        echo -e "${RED}✗ Python 3 not found!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}✓ Activated: $PYTHON_VERSION${NC}"
echo ""

# ============================================================
# Step 3: Install Dependencies
# ============================================================
echo -e "${YELLOW}[3/6] Installing dependencies...${NC}"

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo -e "${YELLOW}Installing from requirements.txt...${NC}"
    pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ requirements.txt not found!${NC}"
    exit 1
fi

# Check PyTorch and GPU
echo -e "${YELLOW}Checking PyTorch and GPU...${NC}"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  Device: Apple Silicon (MPS)')
else:
    print(f'  Device: CPU only')
"
echo ""

# ============================================================
# Step 4: Prepare Output Directory
# ============================================================
echo -e "${YELLOW}[4/6] Preparing output directory...${NC}"

# Create output directories
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/images"
mkdir -p "$PROJECT_ROOT/logs"

# Clean old results if they exist
if [ -f "$RESULTS_FILE" ]; then
    echo -e "${YELLOW}Removing old results...${NC}"
    rm -f "$OUTPUT_DIR"/*.json
    rm -f "$OUTPUT_DIR"/*.txt
    rm -rf "$OUTPUT_DIR/checkpoints"/*
    rm -rf "$OUTPUT_DIR/images"/*
    mkdir -p "$OUTPUT_DIR/checkpoints"
    mkdir -p "$OUTPUT_DIR/images"
fi

echo -e "${GREEN}✓ Output directory ready${NC}"
echo ""

# ============================================================
# Step 5: Run Elliott Wave Pipeline
# ============================================================
echo -e "${YELLOW}[5/6] Running Elliott Wave pattern detection...${NC}"
echo -e "${GREEN}============================================${NC}"

# Show configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Config file: $CONFIG_FILE"
echo -e "  Output: $RESULTS_FILE"
if [ "$USE_HF_DATASET" = true ]; then
    echo -e "  Data source: HuggingFace ($HF_DATASET)"
else
    echo -e "  Data source: yfinance (live)"
fi
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the pipeline
if [ "$USE_HF_DATASET" = true ]; then
    python scripts/pipeline_run.py \
        --use-all-hf-symbols \
        --hf-dataset "$HF_DATASET" \
        --config "$CONFIG_FILE" \
        --output "$RESULTS_FILE" \
        --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
        --resume \
        --verbose
else
    # Use S&P 500 tickers from file
    if [ -f "$PROJECT_ROOT/data/sp500_tickers.txt" ]; then
        SYMBOLS=$(cat "$PROJECT_ROOT/data/sp500_tickers.txt" | tr '\n' ',' | sed 's/,$//')
        python scripts/pipeline_run.py \
            --symbols "$SYMBOLS" \
            --config "$CONFIG_FILE" \
            --output "$RESULTS_FILE" \
            --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
            --resume \
            --verbose
    else
        echo -e "${RED}✗ No symbol file found!${NC}"
        exit 1
    fi
fi

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
RUNTIME_MIN=$((RUNTIME / 60))
RUNTIME_SEC=$((RUNTIME % 60))

echo ""
echo -e "${GREEN}✓ Pattern detection complete!${NC}"
echo -e "${BLUE}  Runtime: ${RUNTIME_MIN}m ${RUNTIME_SEC}s${NC}"
echo ""

# ============================================================
# Step 6: Run Evaluation Scripts
# ============================================================
echo -e "${YELLOW}[6/6] Running evaluation scripts...${NC}"
echo -e "${GREEN}============================================${NC}"

# Check if results exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo -e "${RED}✗ Results file not found: $RESULTS_FILE${NC}"
    exit 1
fi

# Count patterns
PATTERN_COUNT=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(len(data.get('patterns', data)) if isinstance(data, dict) else len(data))")
echo -e "${GREEN}✓ Found $PATTERN_COUNT patterns in results${NC}"
echo ""

# --- EDA Script ---
echo -e "${YELLOW}Running EDA (Exploratory Data Analysis)...${NC}"
if [ -f "$PROJECT_ROOT/tools/evaluate_data_distribution.py" ]; then
    python "$PROJECT_ROOT/tools/evaluate_data_distribution.py" \
        --input "$RESULTS_FILE" \
        --output "$OUTPUT_DIR/eda_report.json"
    
    if [ -f "$OUTPUT_DIR/eda_report.json" ]; then
        echo -e "${GREEN}✓ EDA report saved to: $OUTPUT_DIR/eda_report.json${NC}"
        
        # Print summary
        echo -e "${BLUE}EDA Summary:${NC}"
        python -c "
import json
with open('$OUTPUT_DIR/eda_report.json') as f:
    report = json.load(f)
    print(f\"  Total patterns: {report.get('total_patterns', 'N/A')}\")
    print(f\"  Unique symbols: {report.get('unique_symbols', 'N/A')}\")
    if 'pattern_types' in report:
        print(f\"  Pattern types: {report['pattern_types']}\")
    if 'score_stats' in report:
        stats = report['score_stats']
        print(f\"  Score range: {stats.get('min', 'N/A'):.3f} - {stats.get('max', 'N/A'):.3f}\")
        print(f\"  Mean score: {stats.get('mean', 'N/A'):.3f}\")
"
    fi
else
    echo -e "${YELLOW}⚠ EDA script not found, skipping...${NC}"
fi
echo ""

# --- Rule Validity Script ---
echo -e "${YELLOW}Running pattern validation (Elliott Wave rules)...${NC}"
if [ -f "$PROJECT_ROOT/tools/validate_patterns_v2.py" ]; then
    python "$PROJECT_ROOT/tools/validate_patterns_v2.py" \
        --input "$RESULTS_FILE" \
        --output "$OUTPUT_DIR/validation_report.json"
    
    if [ -f "$OUTPUT_DIR/validation_report.json" ]; then
        echo -e "${GREEN}✓ Validation report saved to: $OUTPUT_DIR/validation_report.json${NC}"
        
        # Print summary
        echo -e "${BLUE}Validation Summary:${NC}"
        python -c "
import json
with open('$OUTPUT_DIR/validation_report.json') as f:
    report = json.load(f)
    total = report.get('total_patterns', 0)
    valid = report.get('valid_patterns', 0)
    rate = (valid / total * 100) if total > 0 else 0
    print(f\"  Total patterns: {total}\")
    print(f\"  Valid patterns: {valid} ({rate:.1f}%)\")
    if 'by_type' in report:
        print(f\"  By type:\")
        for ptype, stats in report['by_type'].items():
            v = stats.get('valid', 0)
            t = stats.get('total', 0)
            r = (v / t * 100) if t > 0 else 0
            print(f\"    {ptype}: {v}/{t} ({r:.1f}%)\")
"
    fi
elif [ -f "$PROJECT_ROOT/tools/evaluate_rule_validity.py" ]; then
    python "$PROJECT_ROOT/tools/evaluate_rule_validity.py" \
        --input "$RESULTS_FILE" \
        --output "$OUTPUT_DIR/validation_report.json"
    echo -e "${GREEN}✓ Validation report saved${NC}"
else
    echo -e "${YELLOW}⚠ Validation script not found, skipping...${NC}"
fi
echo ""

# ============================================================
# Final Summary
# ============================================================
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Pipeline Complete - Final Summary${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${BLUE}Output Files:${NC}"
echo -e "  Results: $RESULTS_FILE"
echo -e "  EDA Report: $OUTPUT_DIR/eda_report.json"
echo -e "  Validation: $OUTPUT_DIR/validation_report.json"
echo -e "  Images: $OUTPUT_DIR/images/"
echo ""
echo -e "${BLUE}Statistics:${NC}"
echo -e "  Patterns detected: $PATTERN_COUNT"
echo -e "  Total runtime: ${RUNTIME_MIN}m ${RUNTIME_SEC}s"
echo ""

# List output files
echo -e "${BLUE}Output directory contents:${NC}"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files)"
echo ""

IMAGE_COUNT=$(ls -1 "$OUTPUT_DIR/images"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo -e "${BLUE}Images generated: $IMAGE_COUNT${NC}"
echo ""
