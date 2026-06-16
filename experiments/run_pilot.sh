#!/usr/bin/env bash
# Local-vs-API comparison pilot.
# Usage: run_pilot.sh <haiku|qwen|local>      (REPS / K overridable via env)
#   haiku -> Anthropic API (claude-haiku-4-5)
#   qwen  -> remote Ollama tunnel (qwen2.5:14b)
#   local -> local HF server (experiments/local_hf_server.py) via the OpenAI provider
#   3 tasks (C wall-assisted, S free-space, S_infeasible free-space)
#   x REPS replicates x K=MAX_ITERATIONS, headless.
# Writes per-cell logs/<model>_<task>_r<rep>.{jsonl,out}.
set -uo pipefail
cd "$(dirname "$0")/.."                      # -> llm_controller/

MODEL_KEY="${1:?usage: run_pilot.sh <haiku|qwen|local>}"
REPS="${REPS:-5}"
K="${K:-40}"
TUNNEL="https://assure-roots-hostels-excessive.trycloudflare.com"
LOCAL_MODEL_ID="${LOCAL_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"   # must match the running local_hf_server
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:8000/v1}"
OVDIR="experiments/_pilot_overlays"
mkdir -p "$OVDIR" logs

# 'local' drives the controller's OpenAI provider against the local HF server.
if [[ "$MODEL_KEY" == "local" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
  export OPENAI_BASE_URL="$LOCAL_BASE_URL"
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$LOCAL_BASE_URL" 2>/dev/null)"
  if [[ -z "$code" || "$code" == "000" ]]; then   # no HTTP response = server down
    echo "ERROR: local model server not reachable at $LOCAL_BASE_URL"
    echo "Start it first (env with torch+transformers), then re-run:"
    echo "  HF_MODEL_ID=$LOCAL_MODEL_ID PORT=8000 HF_DEVICE=cuda:0 \\"
    echo "    /home/andy/miniconda3/envs/moonshot_env/bin/python experiments/local_hf_server.py"
    exit 1
  fi
fi

gen_overlay() {                              # args: task rep -> prints overlay path
  local task="$1" rep="$2" f="$OVDIR/${MODEL_KEY}_${task}_r${rep}.py"
  {
    if [[ "$MODEL_KEY" == "qwen" ]]; then
      echo 'LLM_PROVIDER       = "ollama"'
      echo 'MODEL              = "qwen2.5:14b"'
      echo "OLLAMA_BASE_URL    = \"$TUNNEL\""
      echo 'OLLAMA_TIMEOUT     = 300'
      echo 'OLLAMA_TEMPERATURE = 0.7'        # >0 so replicates differ (config default 0 = identical)
    elif [[ "$MODEL_KEY" == "local" ]]; then
      echo 'LLM_PROVIDER = "openai"'         # -> local HF server via OPENAI_BASE_URL (set above)
      echo "MODEL        = \"$LOCAL_MODEL_ID\""
    else
      echo 'LLM_PROVIDER = "anthropic"'
      echo 'MODEL        = "claude-haiku-4-5"'
    fi
    echo "MAX_ITERATIONS = $K"
    echo 'PLOT_IMAGE     = 0'
    echo "RUN_NAME       = \"${MODEL_KEY}_${task}_r${rep}\""
    [[ "$task" != "C" ]] && echo 'WALLS = []'   # S / S_infeasible are free-space
  } > "$f"
  echo "$f"
}

run_cell() {                                 # args: target task rep [extra cli args...]
  local target="$1" task="$2" rep="$3"; shift 3
  local overlay stem; overlay="$(gen_overlay "$task" "$rep")"
  stem="${MODEL_KEY}_${task}_r${rep}"
  rm -f "logs/${stem}.jsonl" "logs/${stem}.log"
  echo "[$(date +%H:%M:%S)] START $stem"
  MPLBACKEND=Agg timeout 1800 ./venv/bin/python cli.py --target "$target" "$@" --config "$overlay" \
      > "logs/${stem}.out" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $stem (exit $rc) $(grep -hE 'Iterations|Solved' "logs/${stem}.out" | tail -2 | tr '\n' ' ')"
}

echo "[$(date +%H:%M:%S)] PILOT START model=$MODEL_KEY reps=$REPS K=$K"
for rep in $(seq 1 $REPS); do
  run_cell C            C            "$rep" --H 0.15
  run_cell S            S            "$rep"
  run_cell S_infeasible S_infeasible "$rep"
done
echo "[$(date +%H:%M:%S)] PILOT DONE model=$MODEL_KEY"
