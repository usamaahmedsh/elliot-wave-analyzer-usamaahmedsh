"""
Upload a dataset file to Hugging Face Hub under your username.

Usage (recommended):
  1. pip install huggingface_hub git-lfs
  2. huggingface-cli login   # enter your token
  3. git lfs install
  4. python scripts/upload_to_hf.py --file data/all_markets_15y.parquet \
       --repo usamaahmedsh/market-dataset-15y-top200 --private false

The script will:
 - create the repo if it doesn't exist
 - clone it locally (in .hf_tmp_repo)
 - copy the dataset file, README and metadata into the repo
 - commit and push using git + LFS (requires huggingface-cli login or HF_TOKEN env var)

It reads HF token from the environment variable HF_TOKEN if present, otherwise
relies on `huggingface-cli login` having been run.
"""

import argparse
import os
import shutil
from pathlib import Path
import huggingface_hub as _hf
from huggingface_hub import HfApi

# Repository location changed across versions of huggingface_hub; import
# it in a backwards-compatible way and give a helpful error if not found.
Repository = getattr(_hf, 'Repository', None)
if Repository is None:
    try:
        # try submodule (older/newer variations)
        from huggingface_hub.repository import Repository
    except Exception:
        # Don't fail import here; we'll fallback to the HfApi.upload_file path at runtime.
        print("Warning: 'Repository' helper not found in huggingface_hub; falling back to HfApi.upload_file.")
        Repository = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True, help='Path to dataset file (parquet)')
    p.add_argument('--repo', required=True, help='Hugging Face repo id, e.g. username/repo-name')
    p.add_argument('--repo-type', default='dataset', choices=['dataset','model','space'], help='Type of HF repo to create/upload to')
    p.add_argument('--private', default='false', choices=['true','false'], help='Create repo private')
    p.add_argument('--commit-message', default='Add dataset files', help='Git commit message')
    args = p.parse_args()

    src_file = Path(args.file).resolve()
    if not src_file.exists():
        print('File not found:', src_file)
        return 1

    repo_id = args.repo
    repo_type = args.repo_type
    private = args.private.lower() == 'true'

    token = os.environ.get('HF_TOKEN')
    if token:
        print('Using HF_TOKEN from environment')
    else:
        print('HF_TOKEN not found in environment; this script will rely on `huggingface-cli login` credentials.')

    api = HfApi()
    try:
        # attempt to create repo (ignore if exists)
        print('Creating repo', repo_id, 'private=', private, 'type=', repo_type)
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True, token=token)
    except Exception as e:
        print('Warning creating repo (may already exist):', e)

    # If Repository helper is available use git LFS flow; otherwise fallback to HfApi.upload_file
    if 'Repository' in globals() and Repository is not None:
        # clone the repo to a temp folder (Repository will use credentials from huggingface-cli or token)
        tmp_dir = Path('.hf_tmp_repo')
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        print('Cloning repo to', tmp_dir)
        repo = Repository(local_dir=str(tmp_dir), clone_from=repo_id, use_auth_token=token)

        # copy files into the repo
        dest_parquet = tmp_dir / src_file.name
        print('Copying', src_file, '->', dest_parquet)
        shutil.copy(src_file, dest_parquet)

        # copy README and metadata if exist
        readme = Path('data/README_dataset.md')
        meta = Path('data/all_markets_15y_metadata.json')
        if readme.exists():
            shutil.copy(readme, tmp_dir / readme.name)
        if meta.exists():
            shutil.copy(meta, tmp_dir / meta.name)

        # commit & push
        print('Committing and pushing via git LFS...')
        repo.push_to_hub(commit_message=args.commit_message)
        print('Upload complete. Repo available at: https://huggingface.co/' + repo_id)
    else:
        # Fallback: use HfApi.upload_file to push files directly (no git LFS). This works for most sizes
        # but if the file is very large you may prefer the git LFS path locally.
        print('Repository helper not available; using HfApi.upload_file fallback')
        files_to_upload = [src_file]
        readme = Path('data/README_dataset.md')
        meta = Path('data/all_markets_15y_metadata.json')
        if readme.exists():
            files_to_upload.append(readme)
        if meta.exists():
            files_to_upload.append(meta)

        for f in files_to_upload:
            print('Uploading', f.name, 'to', repo_id)
            try:
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                )
            except Exception as e:
                print('Failed to upload', f.name, ':', e)
                return 2
        print('Upload complete. Repo available at: https://huggingface.co/' + repo_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
