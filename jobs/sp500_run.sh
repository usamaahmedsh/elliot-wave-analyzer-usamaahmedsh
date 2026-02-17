#!/bin/bash
#SBATCH --job-name=elliott_sp500
#SBATCH --output=logs/sp500_%j.out
#SBATCH --error=logs/sp500_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# GPU-optimized configuration for 140GB VRAM + 16 cores

# Load required modules (adjust for your HPC cluster)
module load python/3.10
module load cuda/12.1  # Enable CUDA for GPU acceleration

# Print job information
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo "Start time: $(date)"
echo "======================================"
echo ""

# Change to submission directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Activate virtual environment
source venv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check PyTorch and GPU availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else '')"
echo ""

# Create necessary directories
mkdir -p output/checkpoints
mkdir -p logs
mkdir -p output/images

# Count symbols
SYMBOL_COUNT=$(wc -l < data/sp500_tickers.txt)
echo "Processing $SYMBOL_COUNT S&P 500 symbols"
echo ""

# Run the Elliott Wave pipeline
echo "Starting Elliott Wave pattern detection..."
echo "======================================"
python scripts/pipeline_run.py \
  --symbols "$(cat data/sp500_tickers.txt | tr '\n' ',')" \
  --output output/sp500_patterns.json \
  --checkpoint-dir output/checkpoints \
  --resume --verbose 2>&1 | tee logs/pipeline_run_$(date +%Y%m%d_%H%M%S).log

# Check if pipeline succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Pipeline completed successfully!"
    echo "======================================"
    echo ""
    
    # Run data quality validation
    echo "Running data quality validation..."
    python tools/validate_data_quality.py \
      --input output/sp500_patterns.json \
      --output output/sp500_quality_report.txt
    
    # Print summary
    echo ""
    echo "======================================"
    echo "SUMMARY"
    echo "======================================"
    echo "Results saved to: output/sp500_patterns.json"
    echo "Quality report: output/sp500_quality_report.txt"
    echo ""
    
    # Show quality verdict
    if [ -f output/sp500_quality_report.txt ]; then
        echo "Data Quality:"
        grep -A 5 "Quality Verdict" output/sp500_quality_report.txt || echo "Check quality report for details"
    fi
    
    # Show file sizes
    echo ""
    echo "Output files:"
    ls -lh output/sp500_patterns.json
    ls -lh output/sp500_quality_report.txt
    
else
    echo ""
    echo "======================================"
    echo "Pipeline FAILED - check logs for errors"
    echo "======================================"
    echo ""
    echo "To resume from checkpoint:"
    echo "sbatch jobs/sp500_run.sh"
    exit 1
fi

echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "Total walltime: $SECONDS seconds"
echo "======================================"
