"""
Convert a Parquet file to a Hugging Face Dataset and push it to the Hub.

Usage examples:

# interactive login (recommended):
# huggingface-cli login
python scripts/push_parquet_as_dataset.py --file data/all_markets_15y.parquet --repo usamaahmedsh/financial-markets-dataset-15y

# or with HF_TOKEN in env:
export HF_TOKEN=hf_...YOURTOKEN...
python scripts/push_parquet_as_dataset.py --file data/all_markets_15y.parquet --repo usamaahmedsh/financial-markets-dataset-15y

Notes:
- This script creates the HF repo (repo_type='dataset') if it doesn't exist.
- The parquet file will be converted to a single split named 'train' by default.
- For very large files you may prefer the git-LFS flow. This script will still work but may be slower.
"""

from pathlib import Path
import argparse
import os
import pandas as pd
from huggingface_hub import HfApi
from datasets import Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True, help='Path to parquet file')
    p.add_argument('--repo', required=True, help='HF repo id: username/repo-name')
    p.add_argument('--split', default='train', help='Dataset split name (default: train)')
    p.add_argument('--private', action='store_true', help='Create private dataset repo')
    p.add_argument('--chunked', action='store_true', help='If set, stream parquet in chunks (memory-saver)')
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.file)
    if not src.exists():
        print('File not found:', src)
        return 2

    repo_id = args.repo
    repo_type = 'dataset'
    token = os.environ.get('HF_TOKEN')
    if token:
        print('Using HF_TOKEN from environment')
    else:
        print('No HF_TOKEN in env; make sure you ran `huggingface-cli login`')

    api = HfApi()
    try:
        print('Creating repo (if needed):', repo_id)
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=args.private, exist_ok=True, token=token)
    except Exception as e:
        print('Warning creating repo (may already exist):', e)

    # Load parquet into pandas then to datasets.Dataset
    print('Loading parquet (this may use a lot of memory) ->', src)
    try:
        df = pd.read_parquet(src)
    except Exception as e:
        print('Failed to read parquet:', e)
        return 3

    # Ensure Date column is datetime and not an index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Convert to datasets.Dataset
    print('Converting to Hugging Face Dataset (split=', args.split, ')')
    ds = Dataset.from_pandas(df.reset_index(drop=True))

    # push to hub
    target = f"{repo_id}-{args.split}" if not repo_id.endswith(f"-{args.split}") else repo_id
    print('Pushing dataset to hub as repo:', target)
    try:
        ds.push_to_hub(target, private=args.private, token=token)
    except Exception as e:
        print('Failed to push dataset:', e)
        return 4

    print('Push complete. Repo:', target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
