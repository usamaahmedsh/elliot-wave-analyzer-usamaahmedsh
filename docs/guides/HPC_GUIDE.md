# Running on HPC with SLURM

This guide shows how to run the Elliott Wave Analyzer on HPC systems with:
- ✅ SLURM job scheduler
- ✅ Module system (CUDA, Python)
- ✅ Hugging Face dataset integration
- ✅ Checkpoint/resume support
- ✅ Verbose progress tracking

---

## 🚀 Quick Start

### Option 1: Interactive Session
```bash
# Quick test with 3 symbols
./run_hpc.sh AAPL,MSFT,GOOG
```

### Option 2: Batch Job (Recommended)
```bash
# Submit batch job for 10 symbols
sbatch submit_hpc.sh "AAPL,MSFT,GOOG,TSLA,NVDA,META,AMZN,GOOGL,BRK-B,JPM"

# Monitor job
squeue -u $USER

# Check output
tail -f output/slurm-<job_id>.out
```

---

## 📋 HPC Features

### 1. Module System Integration

The scripts automatically load required modules:

```bash
# Python (tries multiple versions)
module load python3/3.10.12  # Preferred
module load python3/3.10.5   # Fallback
module load python3          # Default

# CUDA (tries multiple versions)
module load cuda/12.5        # Latest
module load cuda/12.2        # Fallback
module load cuda/11.8        # Older systems
```

**Your HPC has these available:**
- Python: 3.6.5, 3.7.3, 3.8.10, 3.9.4, **3.10.5**, **3.10.12**, **3.13.8**
- CUDA: 7.5, 8.0, 9.0, 10.0, 11.0, 11.1, 11.2, 11.3, 11.6, **11.8**, **12.2**, **12.5**, 12.8
- PyCUDA: 2019.1

### 2. Hugging Face Dataset

Instead of downloading from Yahoo Finance, data is loaded from your HF dataset:

```python
# Automatic in the pipeline
dataset_name = "usamaahmedsh/financial-markets-dataset-15y-train"

# Benefits:
# - Much faster (no API rate limits)
# - More reliable (cached data)
# - 15 years of historical data
# - 315 symbols across 12 markets
```

### 3. Checkpoint System

Pipeline saves checkpoints after each symbol:

```
output/hpc_batch_TIMESTAMP/
├── checkpoints/
│   ├── processed_symbols.json     # Completed symbols
│   └── partial_results.json       # Accumulated results
├── results.json                   # Final output
└── slurm-<job_id>.out             # Job log
```

**Resume interrupted jobs:**
```bash
./run_pipeline.sh --hpc --resume output/hpc_batch_20260215_143022
```

### 4. Verbose Progress

Real-time progress tracking:

```
[1/10] Processing AAPL...
   Progress: 10.0% complete
   Remaining: 9 symbols
   📊 Data: 3785 bars from 2010-01-04 to 2026-01-15
   🔍 Generated 50 windows
   ⚙️  Running wave analyzer...
   ✅ Found 127 patterns in 8.3s
   📈 Top score: 0.847
   💾 Checkpoint saved
```

---

## 🎯 Usage Examples

### Interactive Jobs

```bash
# Quick test (single symbol)
./run_hpc.sh AAPL

# Multiple symbols
./run_hpc.sh "AAPL,MSFT,GOOG,TSLA,NVDA"

# With full pipeline script
./run_pipeline.sh --hpc --gpu AAPL,MSFT,GOOG
```

### Batch Jobs

**Basic submission:**
```bash
sbatch submit_hpc.sh
```

**Custom symbols:**
```bash
sbatch submit_hpc.sh "AAPL,MSFT,GOOG,TSLA,NVDA,META"
```

**Custom config:**
```bash
sbatch submit_hpc.sh "AAPL,MSFT,GOOG" configs_high_perf.yaml
```

**Large-scale (all 315 symbols):**
```bash
# Create symbol list from dataset
python3 -c "
from datasets import load_dataset
ds = load_dataset('usamaahmedsh/financial-markets-dataset-15y-train', split='train')
symbols = ','.join(sorted(set(ds['ticker'])))
print(symbols)
" > all_symbols.txt

# Submit job
sbatch submit_hpc.sh "$(cat all_symbols.txt)"
```

---

## ⚙️ SLURM Configuration

### Resource Requirements

**Small job (3-10 symbols):**
```bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
```

**Medium job (10-50 symbols):**
```bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8:00:00
```

**Large job (50-315 symbols):**
```bash
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
```

**With GPU:**
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
```

### Job Management

```bash
# Submit job
sbatch submit_hpc.sh

# Check queue
squeue -u $USER

# Cancel job
scancel <job_id>

# View output (while running)
tail -f output/slurm-<job_id>.out

# View completed job info
sacct -j <job_id> --format=JobID,JobName,Elapsed,MaxRSS,CPUTime
```

---

## 📊 Performance Tuning

### CPU Optimization

Edit `configs_high_perf.yaml`:

```yaml
# Match your HPC allocation
processes: 32              # Number of CPU cores allocated
cpu_batch_size: 2048       # Larger batches for HPC
concurrency: 32            # Parallel downloads

# Reduce for faster (but less thorough) runs
max_windows: 1000          # Down from 2000
window_overlap_ratio: 0.25 # Down from 0.5
```

### GPU Optimization

```yaml
use_gpu: true
gpu_batch_size: 20000      # Large batches for HPC GPUs

# Enable GPU in submission
sbatch --gres=gpu:1 submit_hpc.sh
```

### Memory Optimization

If hitting memory limits:

```yaml
# Reduce these in config
max_windows: 500
cpu_batch_size: 512
processes: 16              # Fewer parallel workers
```

---

## 🔍 Monitoring

### Real-time Progress

```bash
# Watch job output
watch -n 5 tail -20 output/slurm-<job_id>.out

# Check resource usage
sstat -j <job_id> --format=JobID,MaxRSS,MaxVMSize,AveCPU
```

### Post-Run Analysis

```bash
# Job efficiency
seff <job_id>

# Detailed accounting
sacct -j <job_id> -o JobID,JobName,Partition,Elapsed,TotalCPU,MaxRSS,State

# GPU usage (if available)
sacct -j <job_id> --format=JobID,ReqGRES,AllocGRES
```

---

## 💾 Data Management

### Hugging Face Cache

Dataset is cached automatically:

```bash
# Cache location (usually)
~/.cache/huggingface/datasets/

# To clear cache
rm -rf ~/.cache/huggingface/datasets/usamaahmedsh___financial-markets-dataset-15y-train
```

### Output Organization

```bash
# Automatic timestamped directories
output/
├── hpc_batch_20260215_100000/
│   ├── results.json
│   ├── checkpoints/
│   ├── evaluation_report.html
│   └── evaluation_metrics.json
├── hpc_batch_20260215_120000/
└── slurm-*.out
```

### Cleanup Old Runs

```bash
# Keep only last 10 runs
ls -dt output/hpc_batch_* | tail -n +11 | xargs rm -rf

# Archive old results
tar -czf archived_results_$(date +%Y%m).tar.gz output/hpc_batch_2026*/
rm -rf output/hpc_batch_202601*
```

---

## 🆘 Troubleshooting

### Module Loading Issues

**Problem:** Module not found
```bash
# Check available modules
module avail python3
module avail cuda

# Load specific version
module load python3/3.10.12
```

**Problem:** Module conflicts
```bash
# Purge all modules first
module purge
module load python3/3.10.12
module load cuda/12.5
```

### Dependency Installation

**Problem:** Permission denied
```bash
# Install to user directory (automatic in scripts)
pip3 install --user -r requirements.txt
```

**Problem:** Slow installation
```bash
# Use HPC pip cache
pip3 install --user --cache-dir=/tmp/$USER/pip -r requirements.txt
```

### Memory Issues

**Problem:** Out of memory
```bash
# Solution 1: Reduce batch size
# Edit configs.yaml:
cpu_batch_size: 256
max_windows: 500

# Solution 2: Request more memory
#SBATCH --mem=128G
```

### GPU Issues

**Problem:** GPU not detected
```bash
# Check GPU availability
nvidia-smi

# Load CUDA module explicitly
module load cuda/12.5

# Verify in job
srun --gres=gpu:1 nvidia-smi
```

**Problem:** CUDA version mismatch
```bash
# Match CUDA module to CuPy version
module load cuda/12.5
pip3 install --user cupy-cuda12x

# Or for CUDA 11
module load cuda/11.8
pip3 install --user cupy-cuda11x
```

### Job Failures

**Problem:** Job times out
```bash
# Resume from checkpoint
./run_pipeline.sh --hpc --resume output/hpc_batch_20260215_143022

# Or increase time limit
#SBATCH --time=48:00:00
```

**Problem:** Job killed by OOM
```bash
# Check memory usage
sacct -j <job_id> --format=JobID,MaxRSS,MaxVMSize

# Increase memory request
#SBATCH --mem=128G
```

---

## 📈 Expected Performance

### Runtime Estimates

| Symbols | CPU Only | GPU Accelerated |
|---------|----------|-----------------|
| 1 | ~30s | ~10s |
| 10 | ~5 min | ~1-2 min |
| 50 | ~25 min | ~5-8 min |
| 100 | ~50 min | ~10-15 min |
| 315 (all) | ~2.5 hrs | ~30-45 min |

*Times vary based on HPC load and configuration*

### Resource Usage

| Symbols | CPU Cores | Memory | GPU VRAM |
|---------|-----------|--------|----------|
| 1-10 | 16 | 16GB | 8GB |
| 10-50 | 32 | 32GB | 16GB |
| 50-100 | 32-64 | 64GB | 24GB |
| 100+ | 64 | 128GB | 40GB |

---

## 🎯 Best Practices

### 1. Start Small
```bash
# Test with 1-3 symbols first
./run_hpc.sh AAPL,MSFT,GOOG

# Then scale up
sbatch submit_hpc.sh "$(head -20 all_symbols.txt | tr '\n' ',')"
```

### 2. Use Checkpoints
```bash
# Always enable checkpointing (automatic)
# If job fails, resume:
./run_pipeline.sh --hpc --resume output/hpc_batch_20260215_143022
```

### 3. Monitor Resources
```bash
# Check job efficiency after completion
seff <job_id>

# Adjust next job based on usage
```

### 4. Batch Processing
```bash
# Process in chunks for better parallelization
# Create multiple jobs for different symbol groups
sbatch submit_hpc.sh "AAPL,MSFT,GOOG"
sbatch submit_hpc.sh "TSLA,NVDA,META"
sbatch submit_hpc.sh "AMZN,GOOGL,BRK-B"
```

---

## 📝 Summary

**HPC features implemented:**
- ✅ Module system integration (Python, CUDA)
- ✅ Hugging Face dataset loading (fast, reliable)
- ✅ Checkpoint/resume support (fault tolerance)
- ✅ Verbose progress tracking (real-time feedback)
- ✅ SLURM batch scripts (automated submission)
- ✅ Resource optimization (CPU/GPU/memory)

**To get started on HPC:**
```bash
# 1. Quick test
./run_hpc.sh AAPL

# 2. Batch job
sbatch submit_hpc.sh

# 3. Monitor
squeue -u $USER

# 4. View results
cat output/hpc_batch_*/results.json | jq '.metadata'
```

**For support:**
- Check SLURM logs: `output/slurm-*.out`
- Review checkpoints: `output/*/checkpoints/`
- Consult HPC docs for your system

---

## 🔗 Related Documentation

- [Automation Guide](AUTOMATION_GUIDE.md) - General pipeline automation
- [GPU Acceleration Guide](GPU_ACCELERATION_GUIDE.md) - GPU setup details
- [Evaluation Quickstart](EVALUATION_QUICKSTART.md) - Result evaluation

Your HPC system is ready to process Elliott Wave patterns at scale! 🚀
