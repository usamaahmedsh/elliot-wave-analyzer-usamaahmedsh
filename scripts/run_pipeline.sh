#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}Running pipeline (setup + execute)...${NC}"

# Ensure venv exists
if [ ! -d ".venv" ]; then
  echo -e "${YELLOW}Creating virtualenv...${NC}"
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
echo -e "${YELLOW}Installing requirements (may be cached)...${NC}"
pip install -r requirements.txt

CFG=configs.yaml
OUTDIR=output
SRC=hf
PROCS=4
GPU_FLAG=""

echo -e "${YELLOW}Using config: ${CFG}${NC}"

echo -e "${YELLOW}Starting pipeline...${NC}"
PYTHONPATH=. python3 scripts/pipeline_run.py AAPL MSFT TSLA --config "$CFG" --source "$SRC" --processes $PROCS --out-dir "$OUTDIR" $GPU_FLAG

echo -e "${GREEN}Pipeline finished. Results in ${OUTDIR}${NC}"

exit 0
