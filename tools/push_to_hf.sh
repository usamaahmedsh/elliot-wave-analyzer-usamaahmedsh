#!/bin/bash
# =============================================================================
# push_to_hf.sh — Upload pipeline output to HuggingFace Datasets
# Usage: bash tools/push_to_hf.sh
# Requires: HUGGING_FACE_HUB_TOKEN already exported in environment
# =============================================================================
set -e

REPO_ID="usamaahmedsh/elliott-wave-patterns"
OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../output" && pwd)"

echo "================================================"
echo "  Elliott Wave → HuggingFace Dataset Upload"
echo "================================================"
echo "  Repo   : $REPO_ID"
echo "  Source : $OUTPUT_DIR"
echo ""

# Check token
if [ -z "$HUGGING_FACE_HUB_TOKEN" ] && [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Neither HUGGING_FACE_HUB_TOKEN nor HF_TOKEN is set."
    exit 1
fi

# Install huggingface_hub if missing
python -c "import huggingface_hub" 2>/dev/null || pip install huggingface_hub --quiet

# Create repo if it doesn't exist yet (safe to run multiple times)
echo "[1/3] Ensuring dataset repo exists..."
python - <<EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN'))
try:
    api.create_repo(repo_id="$REPO_ID", repo_type="dataset", exist_ok=True)
    print("  Repo ready.")
except Exception as e:
    print(f"  Warning: {e}")
EOF

# Upload entire output directory
echo "[2/3] Uploading output/ to HuggingFace..."
python - <<EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN'))
api.upload_folder(
    folder_path="$OUTPUT_DIR",
    repo_id="$REPO_ID",
    repo_type="dataset",
    ignore_patterns=["*.png", "*.jpg", "images/*"],  # skip images to keep size small
    commit_message="Pipeline output upload",
)
print("  Upload complete.")
EOF

# Print result
echo "[3/3] Done!"
echo ""
echo "  Dataset live at:"
echo "  https://huggingface.co/datasets/$REPO_ID"
echo ""
