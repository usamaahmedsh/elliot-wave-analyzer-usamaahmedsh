# Elliott Wave Analyzer - Automation Quick Reference

## 🚀 Getting Started (Choose One)

### Option 1: Fastest (30 seconds)
```bash
./quickstart.sh
```

### Option 2: Full Pipeline
```bash
./run_pipeline.sh
```

### Option 3: Custom Symbols
```bash
./run_pipeline.sh AAPL,MSFT,GOOG,TSLA
```

---

## 📋 Common Commands

| Task | Command |
|------|---------|
| Quick test | `./quickstart.sh` |
| Default run | `./run_pipeline.sh` |
| Custom symbols | `./run_pipeline.sh SYMBOL1,SYMBOL2` |
| High-performance | `./run_pipeline.sh --config configs_high_perf.yaml` |
| With GPU | `./run_pipeline.sh --gpu` |
| Skip evaluation | `./run_pipeline.sh --skip-eval` |
| Quick test + GPU | `./run_pipeline.sh --quick --gpu` |
| Custom output dir | `./run_pipeline.sh --output my_results` |
| Show help | `./run_pipeline.sh --help` |

---

## 📁 Output Files

After running, check `output/TIMESTAMP/`:

| File | What It Contains |
|------|------------------|
| `results.json` | All detected patterns |
| `evaluation_report.html` | Visual report (open in browser) |
| `evaluation_metrics.json` | Metrics (precision, recall, etc.) |
| `summary.txt` | Quick stats and next steps |
| `pipeline.log` | Detailed execution log |

---

## 🔍 Viewing Results

```bash
# Quick summary
cat output/TIMESTAMP/summary.txt

# Open evaluation report
open output/TIMESTAMP/evaluation_report.html

# View all patterns (requires jq)
cat output/TIMESTAMP/results.json | jq '.'

# Count patterns
cat output/TIMESTAMP/results.json | jq '.patterns | length'

# View metrics
cat output/TIMESTAMP/evaluation_metrics.json | jq '.rules.validation_rate'

# Check logs
cat output/TIMESTAMP/pipeline.log
```

---

## ⚡ Performance Modes

| Mode | Command | Time (3 symbols) | Best For |
|------|---------|------------------|----------|
| **Quick Test** | `./quickstart.sh` | ~30-60s | Testing, first run |
| **Standard** | `./run_pipeline.sh` | ~5-10 min | Regular use |
| **High-Perf (CPU)** | `./run_pipeline.sh --config configs_high_perf.yaml` | ~2-4 min | Fast analysis |
| **High-Perf (GPU)** | `./run_pipeline.sh --config configs_high_perf.yaml --gpu` | ~1-2 min | Maximum speed |

---

## 🔧 Configuration Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `configs.yaml` | Default config | Standard runs |
| `configs_high_perf.yaml` | Optimized for speed | Production, large-scale |

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Permission denied | `chmod +x quickstart.sh run_pipeline.sh` |
| Python version error | Install Python 3.9+ |
| GPU not found | Run without `--gpu` or install NVIDIA drivers |
| Out of memory | Reduce `cpu_batch_size` in config |
| Dependency error | Delete `.venv/` and rerun |
| No results | Check `pipeline.log` for errors |

---

## 📚 Full Documentation

- [AUTOMATION_GUIDE.md](doc/AUTOMATION_GUIDE.md) - Complete automation guide
- [AUTOMATION_COMPLETE.md](AUTOMATION_COMPLETE.md) - What was built
- [PIPELINE_FLOW.md](doc/PIPELINE_FLOW.md) - Visual flow diagram

---

## 🎉 One-Line Quick Start

```bash
chmod +x quickstart.sh run_pipeline.sh && ./quickstart.sh
```

**Results in `output/quickstart_results.json` - Everything automatic!**
