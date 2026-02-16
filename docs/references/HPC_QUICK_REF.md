# HPC Quick Reference Card

## 🚀 Quick Commands

```bash
# Quick test (interactive)
./run_hpc.sh AAPL

# Batch job (recommended)
sbatch submit_hpc.sh

# Custom symbols
sbatch submit_hpc.sh "AAPL,MSFT,GOOG,TSLA,NVDA"

# Resume interrupted job
./run_pipeline.sh --hpc --resume output/hpc_batch_TIMESTAMP

# All 315 symbols
sbatch submit_hpc.sh "$(python3 -c 'from datasets import load_dataset; ds = load_dataset(\"usamaahmedsh/financial-markets-dataset-15y-train\", split=\"train\"); print(\",\".join(sorted(set(ds[\"ticker\"]))))')"
```

---

## 📊 Your HPC Resources

**Available modules:**
- Python: 3.10.5, 3.10.12, 3.13.8
- CUDA: 11.8, 12.2, 12.5, 12.8
- PyCUDA: 2019.1

**Loaded automatically by scripts**

---

## 🎯 Expected Performance

| Symbols | Time (CPU) | Time (GPU) |
|---------|-----------|-----------|
| 1 | 30s | 10s |
| 10 | 5 min | 1-2 min |
| 50 | 25 min | 5-8 min |
| 315 (all) | 2.5 hrs | 30-45 min |

---

## 📁 Output Structure

```
output/hpc_batch_20260215_143022/
├── results.json              # Main output
├── checkpoints/
│   ├── processed_symbols.json
│   └── partial_results.json
├── evaluation_report.html
└── evaluation_metrics.json
```

---

## 🔍 Monitoring

```bash
# Check queue
squeue -u $USER

# Watch output
tail -f output/slurm-*.out

# Job stats
seff <job_id>
```

---

## 📚 Documentation

- Complete guide: `doc/HPC_GUIDE.md`
- Summary: `HPC_COMPLETE.md`
- Enhancements: `ENHANCEMENTS_COMPLETE.md`

---

## ✅ Features

- ✅ Auto-loads modules (Python, CUDA)
- ✅ Uses HF dataset (315 symbols, 15 years)
- ✅ Checkpoint/resume (fault-tolerant)
- ✅ Verbose progress (real-time tracking)
- ✅ Batch submission (SLURM ready)

**Run on your HPC cluster with one command!** 🚀
