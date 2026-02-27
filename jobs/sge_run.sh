#!/bin/bash
#$ -N elliott_wave
#$ -o logs/elliott_$JOB_ID.out
#$ -e logs/elliott_$JOB_ID.err
#$ -l h_rt=6:00:00
#$ -l h_vmem=32G
#$ -pe smp 16
#$ -l gpu=1
#$ -cwd
#$ -V

# ============================================================
# Elliott Wave Analyzer - SGE Job Script
# ============================================================
# Submit with: qsub jobs/sge_run.sh
# Interactive:  qrsh -l h_vmem=32G -pe smp 16 -l gpu=1
# ============================================================

# Load required modules
module load cuda/12.2
module load python3/3.12.4

# Print job information
echo "======================================"
echo "Job ID: $JOB_ID"
echo "Job Name: $JOB_NAME"
echo "Node: $(hostname)"
echo "CPUs: $NSLOTS"
echo "Memory: 32GB"
echo "GPU: 12GB VRAM"
echo "Start time: $(date)"
echo "======================================"
echo ""

# Run the pipeline
./run_pipeline.sh

echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"
