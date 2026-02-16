#!/bin/bash
# Elliott Wave Analyzer - Fully Automated End-to-End Pipeline
# 
# This script runs the complete pipeline from scratch:
# 1. Environment setup (HPC modules or venv)
# 2. Data loading from Hugging Face
# 3. Pattern detection (with checkpoints)
# 4. Evaluation
# 5. Results storage
#
# Usage:
#   ./run_pipeline.sh                           # Run with defaults
#   ./run_pipeline.sh AAPL,MSFT,GOOG            # Custom symbols
#   ./run_pipeline.sh --config configs_high_perf.yaml  # Custom config
#   ./run_pipeline.sh --hpc                     # Use HPC modules
#   ./run_pipeline.sh --resume CHECKPOINT_DIR   # Resume from checkpoint
#   ./run_pipeline.sh --help                    # Show help

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default configuration
SYMBOLS="${1:-AAPL,MSFT,GOOG}"
CONFIG_FILE="configs.yaml"
OUTPUT_DIR="output/$(date +%Y%m%d_%H%M%S)"
VENV_DIR=".venv"
PYTHON_VERSION="3.11"
HF_DATASET="usamaahmedsh/financial-markets-dataset-15y-train"

# Parse arguments
USE_HPC=false
INSTALL_GPU=false
SKIP_EVALUATION=false
QUICK_MODE=false
RESUME_FROM=""
VERBOSE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --hpc)
            USE_HPC=true
            shift
            ;;
        --gpu)
            INSTALL_GPU=true
            shift
            ;;
        --skip-eval)
            SKIP_EVALUATION=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            SYMBOLS="AAPL"
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [SYMBOLS] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  SYMBOLS              Comma-separated symbols (default: AAPL,MSFT,GOOG)"
            echo ""
            echo "Options:"
            echo "  --config FILE        Config file to use (default: configs.yaml)"
            echo "  --output DIR         Output directory (default: output/TIMESTAMP)"
            echo "  --hpc                Use HPC modules instead of venv"
            echo "  --gpu                Install GPU acceleration (CuPy)"
            echo "  --skip-eval          Skip evaluation step"
            echo "  --quick              Quick mode (single symbol AAPL)"
            echo "  --resume DIR         Resume from checkpoint directory"
            echo "  --quiet              Disable verbose progress"
            echo "  --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 --hpc                              # Use HPC modules"
            echo "  $0 AAPL,MSFT --config configs_high_perf.yaml"
            echo "  $0 --resume output/20260215_143022    # Resume from checkpoint"
            echo "  $0 --hpc --gpu                        # HPC with GPU"
            exit 0
            ;;
        *)
            SYMBOLS="$1"
            shift
            ;;
    esac
done

# ============================================================================
# Logging
# ============================================================================

LOG_FILE="$OUTPUT_DIR/pipeline.log"
mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "============================================================================" | tee -a "$LOG_FILE"
}

log_section "ELLIOTT WAVE ANALYZER - AUTOMATED PIPELINE"
log "Project root: $PROJECT_ROOT"
log "Output directory: $OUTPUT_DIR"
log "Config file: $CONFIG_FILE"
log "Symbols: $SYMBOLS"
log "HPC mode: $USE_HPC"
log "Verbose: $VERBOSE"
if [ -n "$RESUME_FROM" ]; then
    log "Resuming from: $RESUME_FROM"
fi

# ============================================================================
# Step 1: Check Python Version
# ============================================================================

log_section "STEP 1: Checking Python Version"

if ! command -v python3 &> /dev/null; then
    log "ERROR: python3 not found. Please install Python 3.11+"
    exit 1
fi

PYTHON_VER=$(python3 --version | awk '{print $2}')
log "Found Python: $PYTHON_VER"

# Check if version is at least 3.9
PYTHON_MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VER | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    log "ERROR: Python 3.9+ required. Found: $PYTHON_VER"
    exit 1
fi

log "✅ Python version OK"

# ============================================================================
# Step 2: Setup Environment (HPC or Virtual Environment)
# ============================================================================

log_section "STEP 2: Setting Up Environment"

if [ "$USE_HPC" = true ]; then
    log "Using HPC module system..."
    
    # Load Python module
    if command -v module &> /dev/null; then
        log "Loading Python module..."
        
        # Try to load Python 3.10+ (available on your HPC)
        if module avail python3/3.10.12 2>&1 | grep -q "python3/3.10.12"; then
            module load python3/3.10.12
            log "✅ Loaded python3/3.10.12"
        elif module avail python3/3.10.5 2>&1 | grep -q "python3/3.10.5"; then
            module load python3/3.10.5
            log "✅ Loaded python3/3.10.5"
        elif module avail python3/3.13.8 2>&1 | grep -q "python3/3.13.8"; then
            module load python3/3.13.8
            log "✅ Loaded python3/3.13.8"
        else
            log "WARNING: Could not find Python 3.10+, trying default..."
            module load python3
            log "✅ Loaded python3 (default)"
        fi
        
        # Load CUDA module if GPU requested
        if [ "$INSTALL_GPU" = true ]; then
            log "Loading CUDA module..."
            if module avail cuda/12.5 2>&1 | grep -q "cuda/12.5"; then
                module load cuda/12.5
                log "✅ Loaded cuda/12.5"
            elif module avail cuda/12.2 2>&1 | grep -q "cuda/12.2"; then
                module load cuda/12.2
                log "✅ Loaded cuda/12.2"
            elif module avail cuda/11.8 2>&1 | grep -q "cuda/11.8"; then
                module load cuda/11.8
                log "✅ Loaded cuda/11.8"
            else
                log "WARNING: Could not find CUDA 11.8+, trying default..."
                module load cuda
                log "✅ Loaded cuda (default)"
            fi
            
            # Optionally load pycuda
            if module avail pycuda/2019.1 2>&1 | grep -q "pycuda/2019.1"; then
                module load pycuda/2019.1
                log "✅ Loaded pycuda/2019.1"
            fi
        fi
        
        # Show loaded modules
        log "Loaded modules:"
        module list 2>&1 | tee -a "$LOG_FILE"
        
    else
        log "ERROR: module command not found. Not running on HPC?"
        log "Try running without --hpc flag"
        exit 1
    fi
    
    # Verify Python version
    PYTHON_CMD="python3"
    PYTHON_VER=$(python3 --version | awk '{print $2}')
    log "Python version: $PYTHON_VER"
    
else
    # Original virtual environment setup
    log "Using virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        log "Virtual environment already exists at: $VENV_DIR"
        log "Using existing environment..."
    else
        log "Creating virtual environment at: $VENV_DIR"
        python3 -m venv "$VENV_DIR"
        log "✅ Virtual environment created"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log "✅ Virtual environment activated"

    # Upgrade pip
    log "Upgrading pip..."
    python -m pip install --upgrade pip --quiet
    log "✅ pip upgraded"
    
    PYTHON_CMD="python"
fi

# ============================================================================
# Step 3: Install Dependencies
# ============================================================================

log_section "STEP 3: Installing Dependencies"

if [ "$USE_HPC" = true ]; then
    log "HPC mode: Installing to user directory..."
    
    # Install packages to user directory
    PIP_CMD="pip3 install --user"
    
    if [ -f "requirements.txt" ]; then
        log "Installing packages from requirements.txt..."
        $PIP_CMD -r requirements.txt --quiet 2>&1 | tee -a "$LOG_FILE" || {
            log "Installing with pip install --user (without quiet)..."
            $PIP_CMD -r requirements.txt 2>&1 | tee -a "$LOG_FILE"
        }
        log "✅ Dependencies installed to user directory"
    else
        log "ERROR: requirements.txt not found"
        exit 1
    fi
    
else
    # Original venv installation
    if [ -f "requirements.txt" ]; then
        log "Installing packages from requirements.txt..."
        
        # Check if requirements are already installed
        if pip freeze | grep -q "numpy"; then
            log "Dependencies appear to be installed. Checking for updates..."
            pip install -r requirements.txt --upgrade --quiet 2>&1 | tee -a "$LOG_FILE"
        else
            log "Installing fresh dependencies..."
            pip install -r requirements.txt 2>&1 | tee -a "$LOG_FILE"
        fi
        
        log "✅ Dependencies installed"
    else
        log "ERROR: requirements.txt not found"
        exit 1
    fi
fi

# ============================================================================
# Step 4: Install GPU Acceleration (Optional)
# ============================================================================

if [ "$INSTALL_GPU" = true ]; then
    log_section "STEP 4: Installing GPU Acceleration"
    
    if [ "$USE_HPC" = true ]; then
        log "HPC mode: CUDA module already loaded"
        log "Installing CuPy to user directory..."
        
        # Detect CUDA version from loaded module
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/')
            log "CUDA version: $CUDA_VERSION"
        else
            CUDA_VERSION="12.5"
            log "Could not detect CUDA version, assuming 12.5"
        fi
        
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        if [ "$CUDA_MAJOR" = "12" ]; then
            log "Installing cupy-cuda12x..."
            pip3 install --user cupy-cuda12x 2>&1 | tee -a "$LOG_FILE"
        elif [ "$CUDA_MAJOR" = "11" ]; then
            log "Installing cupy-cuda11x..."
            pip3 install --user cupy-cuda11x 2>&1 | tee -a "$LOG_FILE"
        else
            log "WARNING: Unsupported CUDA version $CUDA_VERSION. Attempting cupy-cuda12x..."
            pip3 install --user cupy-cuda12x 2>&1 | tee -a "$LOG_FILE"
        fi
        
    elif command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected"
        
        # Detect CUDA version
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/')
            log "CUDA version: $CUDA_VERSION"
        else
            CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1.\2/' || echo "12")
            log "CUDA version (from nvidia-smi): $CUDA_VERSION"
        fi
        
        # Install appropriate CuPy version
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        if [ "$CUDA_MAJOR" = "12" ]; then
            log "Installing cupy-cuda12x..."
            pip install cupy-cuda12x 2>&1 | tee -a "$LOG_FILE"
        elif [ "$CUDA_MAJOR" = "11" ]; then
            log "Installing cupy-cuda11x..."
            pip install cupy-cuda11x 2>&1 | tee -a "$LOG_FILE"
        else
            log "WARNING: Unsupported CUDA version $CUDA_VERSION. Attempting cupy-cuda12x..."
            pip install cupy-cuda12x 2>&1 | tee -a "$LOG_FILE"
        fi
    else
        log "⚠️  No NVIDIA GPU detected. Skipping GPU installation."
    fi
    
    # Verify installation
    if $PYTHON_CMD -c "import cupy; print('CuPy OK')" 2>&1 | grep -q "CuPy OK"; then
        log "✅ GPU acceleration installed successfully"
    else
        log "⚠️  GPU acceleration installation failed. Continuing with CPU-only..."
    fi
else
    log_section "STEP 4: GPU Acceleration (Skipped)"
    log "Use --gpu flag to enable GPU acceleration"
fi

# ============================================================================
# Step 5: Verify Installation
# ============================================================================

log_section "STEP 5: Verifying Installation"

log "Checking key dependencies..."

# Check critical packages
PACKAGES=("numpy" "pandas" "numba" "yfinance")
ALL_OK=true

for pkg in "${PACKAGES[@]}"; do
    if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        VERSION=$($PYTHON_CMD -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        log "  ✅ $pkg ($VERSION)"
    else
        log "  ❌ $pkg - NOT INSTALLED"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    log "ERROR: Some dependencies are missing. Please check installation."
    exit 1
fi

log "✅ All dependencies verified"

# ============================================================================
# Step 6: Pre-warm Numba JIT
# ============================================================================

log_section "STEP 6: Pre-warming Numba JIT"

log "Compiling Numba functions (first-time compilation)..."
if $PYTHON_CMD -c "from pipeline.numba_warm import prewarm_numba; prewarm_numba()" 2>&1 | tee -a "$LOG_FILE"; then
    log "✅ Numba JIT pre-warmed"
else
    log "⚠️  Numba pre-warming failed (non-critical)"
fi

# ============================================================================
# Step 7: Run Pipeline
# ============================================================================

log_section "STEP 7: Running Pattern Detection Pipeline"

# Check if resuming from checkpoint
if [ -n "$RESUME_FROM" ]; then
    log "Resuming from checkpoint: $RESUME_FROM"
    RESULTS_FILE="$RESUME_FROM/results.json"
    CHECKPOINT_DIR="$RESUME_FROM/checkpoints"
    
    # Use same output directory
    OUTPUT_DIR="$RESUME_FROM"
    LOG_FILE="$OUTPUT_DIR/pipeline.log"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        log "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
        exit 1
    fi
else
    RESULTS_FILE="$OUTPUT_DIR/results.json"
    CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
    mkdir -p "$CHECKPOINT_DIR"
fi

log "Running pipeline with:"
log "  Symbols: $SYMBOLS"
log "  Config: $CONFIG_FILE"
log "  Output: $RESULTS_FILE"
log "  Checkpoint dir: $CHECKPOINT_DIR"
log "  HF Dataset: $HF_DATASET"
log "  Verbose: $VERBOSE"

# Update config to use correct output directory
TEMP_CONFIG="$OUTPUT_DIR/temp_config.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Set environment variables for the pipeline
export PIPELINE_OUTPUT_DIR="$OUTPUT_DIR"
export PIPELINE_CHECKPOINT_DIR="$CHECKPOINT_DIR"
export HF_DATASET_NAME="$HF_DATASET"
export PIPELINE_VERBOSE="$VERBOSE"
if [ -n "$RESUME_FROM" ]; then
    export PIPELINE_RESUME="true"
else
    export PIPELINE_RESUME="false"
fi

# Run pipeline
log "Starting pattern detection..."
START_TIME=$(date +%s)

if $PYTHON_CMD scripts/pipeline_run.py \
    --symbols "$SYMBOLS" \
    --config "$TEMP_CONFIG" \
    --output "$RESULTS_FILE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --hf-dataset "$HF_DATASET" \
    $([ "$VERBOSE" = true ] && echo "--verbose" || echo "") \
    $([ -n "$RESUME_FROM" ] && echo "--resume" || echo "") \
    2>&1 | tee -a "$LOG_FILE"; then
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    log "✅ Pipeline completed in ${ELAPSED}s"
else
    log "❌ Pipeline failed. Check log: $LOG_FILE"
    exit 1
fi

# ============================================================================
# Step 8: Evaluate Results
# ============================================================================

if [ "$SKIP_EVALUATION" = false ]; then
    log_section "STEP 8: Evaluating Results"
    
    EVAL_REPORT="$OUTPUT_DIR/evaluation_report.html"
    EVAL_JSON="$OUTPUT_DIR/evaluation_metrics.json"
    
    log "Running evaluation..."
    
    if [ -f "$RESULTS_FILE" ]; then
        # Run rule compliance validation
        log "Checking rule compliance..."
        if $PYTHON_CMD utils/evaluation/evaluate_patterns.py \
            --mode all \
            --input "$RESULTS_FILE" \
            --report "$EVAL_REPORT" \
            --output "$EVAL_JSON" \
            2>&1 | tee -a "$LOG_FILE"; then
            
            log "✅ Evaluation completed"
            log "  Report: $EVAL_REPORT"
            log "  Metrics: $EVAL_JSON"
        else
            log "⚠️  Evaluation failed (non-critical)"
        fi
    else
        log "⚠️  No results file found. Skipping evaluation."
    fi
else
    log_section "STEP 8: Evaluation (Skipped)"
fi

# ============================================================================
# Step 9: Generate Summary
# ============================================================================

log_section "STEP 9: Generating Summary"

SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

cat > "$SUMMARY_FILE" << EOF
ELLIOTT WAVE ANALYZER - PIPELINE SUMMARY
========================================

Run Date: $(date)
Symbols: $SYMBOLS
Config: $CONFIG_FILE

OUTPUT FILES
------------
Results:     $RESULTS_FILE
Evaluation:  $EVAL_REPORT
Metrics:     $EVAL_JSON
Log:         $LOG_FILE

QUICK STATS
-----------
EOF

# Extract quick stats from results if available
if [ -f "$RESULTS_FILE" ]; then
    # Count patterns (simple grep approach)
    PATTERN_COUNT=$(grep -o '"pattern"' "$RESULTS_FILE" 2>/dev/null | wc -l || echo "0")
    echo "Patterns Detected: $PATTERN_COUNT" >> "$SUMMARY_FILE"
fi

if [ -f "$EVAL_JSON" ]; then
    # Extract metrics if available
    if command -v jq &> /dev/null; then
        echo "" >> "$SUMMARY_FILE"
        echo "EVALUATION METRICS" >> "$SUMMARY_FILE"
        echo "------------------" >> "$SUMMARY_FILE"
        jq -r '.rules.validation_rate // "N/A"' "$EVAL_JSON" 2>/dev/null | \
            xargs printf "Rule Compliance: %s\n" >> "$SUMMARY_FILE" || true
    fi
fi

cat >> "$SUMMARY_FILE" << EOF

NEXT STEPS
----------
1. View results:     cat $RESULTS_FILE
2. View evaluation:  open $EVAL_REPORT
3. Check logs:       cat $LOG_FILE

For detailed analysis, see the evaluation report.
EOF

cat "$SUMMARY_FILE" | tee -a "$LOG_FILE"

# ============================================================================
# Step 10: Cleanup and Final Status
# ============================================================================

log_section "STEP 10: Pipeline Complete"

log "All outputs saved to: $OUTPUT_DIR"
log ""
log "📁 Output Files:"
log "  ├─ results.json              - Detected patterns"
log "  ├─ evaluation_report.html    - Evaluation report"
log "  ├─ evaluation_metrics.json   - Metrics data"
log "  ├─ summary.txt               - Quick summary"
log "  └─ pipeline.log              - Detailed log"
log ""

if [ "$SKIP_EVALUATION" = false ] && [ -f "$EVAL_REPORT" ]; then
    log "🎉 SUCCESS! Pipeline completed successfully."
    log ""
    log "Next steps:"
    log "  1. View evaluation report:"
    log "     open $EVAL_REPORT"
    log ""
    log "  2. Analyze results:"
    log "     cat $RESULTS_FILE | jq '.'"
    log ""
    log "  3. Review metrics:"
    log "     cat $EVAL_JSON | jq '.'"
else
    log "✅ Pipeline completed successfully."
    log ""
    log "Next steps:"
    log "  1. View results:"
    log "     cat $RESULTS_FILE"
    log ""
    log "  2. Run evaluation manually:"
    log "     python scripts/evaluate_patterns.py --mode all --input $RESULTS_FILE"
fi

log ""
log "============================================================================"

# Return to original directory
cd - > /dev/null 2>&1 || true

exit 0
