#!/bin/bash
#SBATCH --job-name=elliott-wave
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=output/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Elliott Wave Analyzer - SLURM Batch Script
# 
# Submit with: sbatch submit_hpc.sh
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>

echo "=========================================="
echo "Elliott Wave Analyzer - HPC Batch Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load modules
echo "Loading modules..."
module load python3/3.10.12 || module load python3/3.10.5 || module load python3
module load cuda/12.5 || module load cuda/12.2 || module load cuda

echo ""
echo "Loaded modules:"
module list
echo ""

# Install dependencies (first time only)
echo "Installing dependencies..."
pip3 install --user -r requirements.txt --quiet 2>&1 || pip3 install --user -r requirements.txt

# Create output directory
OUTPUT_DIR="output/hpc_batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Configuration
SYMBOLS="${1:-AAPL,MSFT,GOOG,TSLA,NVDA,META,AMZN,GOOGL,BRK-B,JPM}"
CONFIG="${2:-configs_high_perf.yaml}"
HF_DATASET="usamaahmedsh/financial-markets-dataset-15y-train"

echo "Configuration:"
echo "  Symbols: $SYMBOLS"
echo "  Config: $CONFIG"
echo "  Dataset: $HF_DATASET"
echo "  Output: $OUTPUT_DIR"
echo ""

# Set environment
export PYTHONPATH=.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run pipeline
echo "Starting pipeline..."
echo ""

python3 scripts/pipeline_run_enhanced.py \
    --symbols "$SYMBOLS" \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR/results.json" \
    --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
    --hf-dataset "$HF_DATASET" \
    --verbose

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Pipeline completed successfully!"
    echo "=========================================="
    echo "Results: $OUTPUT_DIR/results.json"
    echo "End time: $(date)"
    echo ""
    
    # Run evaluation
    echo "Running evaluation..."
    python3 utils/evaluation/evaluate_patterns.py \
        --mode all \
        --input "$OUTPUT_DIR/results.json" \
        --report "$OUTPUT_DIR/evaluation_report.html" \
        --output "$OUTPUT_DIR/evaluation_metrics.json"
    
    echo ""
    echo "Evaluation complete!"
    echo "Report: $OUTPUT_DIR/evaluation_report.html"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Pipeline failed!"
    echo "=========================================="
    echo "Check logs: $OUTPUT_DIR/checkpoints/"
    echo ""
fi

# Print resource usage
echo "Resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,CPUTime

echo ""
echo "Job completed at: $(date)"
