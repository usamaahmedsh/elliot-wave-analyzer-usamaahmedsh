#!/usr/bin/env python3
"""Sweep cpu_batch_size x cpu_top_k for the pipeline on GOOG and record timings/stats.

This script imports the pipeline runner and calls run_pipeline programmatically so
we can set cfg values per-run without editing YAML repeatedly.
"""
import asyncio
import time
import csv
import glob
from pathlib import Path

from pipeline.config import PipelineConfig
from scripts.pipeline_run import run_pipeline
import yaml
from pathlib import Path as _P

OUT_CSV = 'output/cpu_sweep_results.csv'
Path('output').mkdir(exist_ok=True)

def main():
    # Attempt to read sweep ranges from configs.yaml (so you can tune them there)
    cfg_path = _P('configs.yaml')
    batch_sizes = [128, 256, 512, 1024]
    top_ks = [16, 32, 64, 128]
    if cfg_path.exists():
        try:
            with cfg_path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            if 'cpu_batch_sizes' in data and isinstance(data['cpu_batch_sizes'], list):
                batch_sizes = [int(x) for x in data['cpu_batch_sizes']]
            if 'cpu_top_k_values' in data and isinstance(data['cpu_top_k_values'], list):
                top_ks = [int(x) for x in data['cpu_top_k_values']]
        except Exception:
            pass

    rows = []
    for b in batch_sizes:
        for k in top_ks:
            print(f"Running sweep: batch_size={b} top_k={k}")
            cfg = PipelineConfig.load_from_file('configs.yaml')
            cfg.cpu_batch_size = b
            cfg.cpu_top_k = k
            cfg.processes = cfg.processes or 4
            cfg.profile = True
            cfg.days = 365

            t0 = time.time()
            results = asyncio.run(run_pipeline(['GOOG'], cfg))
            elapsed = time.time() - t0

            # collect stats from returned results
            n_pre = 0
            n_full = 0
            try:
                geo = results.get('GOOG')
                if geo:
                    for r in geo:
                        st = r.get('scan_stats')
                        if st:
                            n_pre += int(st.get('n_pre_scored', 0))
                            n_full += int(st.get('n_full_evals', 0))
            except Exception:
                pass

            # find most recent data/results_run_*.json
            matches = sorted(glob.glob('data/results_run_*.json'))
            latest = matches[-1] if matches else ''
            rows.append({'batch_size': b, 'top_k': k, 'elapsed_total': elapsed, 'n_pre_scored': n_pre, 'n_full_evals': n_full, 'results_path': latest})

            # move latest results to output for traceability
            if latest:
                dst = Path('output') / f'results_run_batch{b}_top{k}.json'
                try:
                    Path(latest).rename(dst)
                except Exception:
                    try:
                        import shutil
                        shutil.copy(latest, dst)
                    except Exception:
                        pass

            # persist partial CSV after each run
            with open(OUT_CSV, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print('Sweep complete. Results written to', OUT_CSV)


if __name__ == '__main__':
    main()
