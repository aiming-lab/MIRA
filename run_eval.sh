#!/bin/bash
# Full pipeline: download MIRA -> evaluate -> accuracy with LLM judge -> save to results.txt
#
# Usage:
#   ./run_eval.sh              # evaluate ALL models in model_config.py
#   ./run_eval.sh -m gpt4o    # evaluate a single model
#
# Required: in model_config.py
#   - MODEL_CONFIG: api_key, api_version, azure_endpoint, model_name (for eval)
#   - JUDGE_CONFIG: api_key, api_version, azure_endpoint, model_name (for acc --use-llm-judge)
#   - Optional: AZURE_OPENAI_API_KEY env overrides JUDGE_CONFIG["api_key"]
#
# Prereq: pip install -r requirements.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MIRA_DIR="./MIRA"
EVAL_OUTPUT_DIR="./eval_output"
RESULTS_FILE="./results.txt"

# Parse optional -m MODEL_NAME argument
MODEL_FLAG=""
while getopts "m:" opt; do
  case "$opt" in
    m) MODEL_FLAG="-m $OPTARG" ;;
    *) echo "Usage: $0 [-m MODEL_NAME]"; exit 1 ;;
  esac
done

echo "========== 0. Check dependencies =========="
python -c "import huggingface_hub, openai, tqdm, PIL" || { echo "Run: pip install -r requirements.txt"; exit 1; }

echo ""
echo "========== 1. Download MIRA dataset (all tasks, 60s between tasks for rate limits) =========="
python download_data.py

echo ""
echo "========== 2. Evaluate ${MODEL_FLAG:-all models} (8 workers, direct_answer + text_cot + visual_cot) =========="
echo "         (loop until complete: re-run on crash or if any response is API_CALL_FAILED or empty)"
while true; do
  ret=0
  python eval_azure_api.py \
    -b "$MIRA_DIR" \
    -o "$EVAL_OUTPUT_DIR" \
    -w 8 \
    $MODEL_FLAG || ret=$?

  if [ "$ret" -ne 0 ]; then
    echo "eval_azure_api exited $ret, re-running in 10s..."
    sleep 10
    continue
  fi

  failed=0
  for f in "$EVAL_OUTPUT_DIR"/*.jsonl; do
    [ -f "$f" ] || continue
    if grep -q 'API_CALL_FAILED' "$f" 2>/dev/null || grep -qE '"response"[[:space:]]*:[[:space:]]*""' "$f" 2>/dev/null; then
      failed=1
      break
    fi
  done
  if [ "$failed" -eq 1 ]; then
    echo "Some responses are API_CALL_FAILED or empty, re-running eval in 10s..."
    sleep 10
    continue
  fi
  break
done

echo ""
echo "========== 3. Accuracy with LLM judge (1 worker), save to $RESULTS_FILE =========="
python acc.py \
  -r "$EVAL_OUTPUT_DIR" \
  --use-llm-judge \
  -w 1 \
  2>&1 | tee "$RESULTS_FILE"

echo ""
echo "Done. Full results saved to $RESULTS_FILE"
