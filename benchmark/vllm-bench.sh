#!/usr/bin/env bash
# vllm-bench.sh: repeatable load test for the vLLM container.
#
#   ./vllm-bench.sh <label> [--force]
#   e.g.   ./vllm-bench.sh baseline
# Compare two runs: python3 vllm-bench-compare.py <label-a> <label-b>
#
# Runs vLLM's own benchmark client (vllm bench serve) inside the container
# against the running server, so nothing new gets installed. Results are written
# inside the container, then copied out to ./benchmark/results/<label>/.
#
# Env overrides: 
# REPS (default 3)
# CONTAINER (default flint)
# SEED (default 42)
# SERVED_NAME (default flint)
# MODEL_ID (default /model)
set -euo pipefail

LABEL="${1:-}"
FORCE="${2:-}"
CONTAINER="${CONTAINER:-flint}"
REPS="${REPS:-3}"
SEED="${SEED:-42}"
MODEL_ID="${MODEL_ID:-/model}"                        # model is mounted at /model in the container; used for the tokenizer
SERVED_NAME="${SERVED_NAME:-flint}"                    # must match --served-model-name
RESULTS_ROOT="./benchmark/results"
CTR_BENCH_ROOT="/root/.cache/huggingface/bench"

# Scenarios: name  input_tokens  output_tokens  "concurrency levels"
#   chat:    typical Open WebUI turn (RAG context + answer); decode heavy
#   prefill: long document in, short answer out; all-reduce heavy prefill
SCENARIOS=(
  "chat    2000  400 1 4 8 16"
  "prefill 16000 200 1 4"
)

if [[ -z "$LABEL" ]]; then
  echo "usage: $0 <label> [--force]" >&2
  exit 1
fi

OUT_HOST="$RESULTS_ROOT/$LABEL"
OUT_CTR="$CTR_BENCH_ROOT/$LABEL"
if [[ -e "$OUT_HOST" ]]; then
  echo "Results for '$LABEL' already exist in $OUT_HOST. Pick a new label." >&2
  exit 1
fi

# runs without auth; the key is optional and only used if present.
KEY="$(docker exec "$CONTAINER" printenv VLLM_API_KEY || true)"

# --- Refuse to run while real users are on the box (it skews everything) ---
busy="$(docker exec -e K="$KEY" "$CONTAINER" python3 -c '
import os, urllib.request
k = os.environ.get("K", "")
headers = {"Authorization": "Bearer " + k} if k else {}
req = urllib.request.Request("http://localhost:8000/metrics", headers=headers)
n = 0.0
for line in urllib.request.urlopen(req).read().decode().splitlines():
    if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
        n += float(line.rsplit(" ", 1)[1])
print(int(n))
')"
if [[ "$busy" != "0" && "$FORCE" != "--force" ]]; then
  echo "vLLM has $busy request(s) in flight from real users. Run when idle, or pass --force." >&2
  exit 1
fi

mkdir -p "$OUT_HOST"

# --- Record the conditions so runs can be checked for fairness ---
{
  echo "label:      $LABEL"
  echo "date:       $(date -Is)"
  echo "image:      $(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
  echo "reps:       $REPS   seed: $SEED"
  echo
  echo "== vLLM all-reduce / backend path (from container log) =="
  docker logs "$CONTAINER" 2>&1 | grep -Ei "allreduce|all-reduce|ccl|sycl|reduce-scatter" | tail -n 5 || echo "(no matching lines)"
} > "$OUT_HOST/conditions.txt"

bench() {  # bench <input> <output> <concurrency> <num_prompts> <filename|"">
  local args=(
    --backend openai --endpoint /v1/completions
    --base-url http://localhost:8000
    --model "$MODEL_ID" --served-model-name "$SERVED_NAME"
    --dataset-name random
    --random-input-len "$1" --random-output-len "$2"
    --max-concurrency "$3" --num-prompts "$4"
    --seed "$SEED" --temperature 0 --ignore-eos
  )
  if [[ -n "$5" ]]; then
    args+=(--save-result --result-dir "$OUT_CTR" --result-filename "$5")
  fi
  if [[ -n "$KEY" ]]; then
    docker exec -e OPENAI_API_KEY="$KEY" "$CONTAINER" vllm bench serve "${args[@]}"
  else
    docker exec "$CONTAINER" vllm bench serve "${args[@]}"
  fi
}

# --- Warmup (not saved): triggers the one-time Triton JIT compiles ---
docker exec "$CONTAINER" mkdir -p "$OUT_CTR"
echo ">>> warmup"
bench 2000 200 8 16 "" > /dev/null

# --- Measured runs ---
for rep in $(seq 1 "$REPS"); do
  for line in "${SCENARIOS[@]}"; do
    read -r name in_len out_len concs <<< "$line"
    for c in $concs; do
	prompts=$(( c * (c >= 4 ? 4 : 8) ))
      file="${name}_c${c}_r${rep}.json"
      echo ">>> rep $rep/$REPS  $name  in=$in_len out=$out_len  concurrency=$c  prompts=$prompts"
      bench "$in_len" "$out_len" "$c" "$prompts" "$file" \
        | grep -E "Output token throughput|Median TTFT|Median TPOT|Failed requests" || true
    done
  done
done

# --- Copy results out of the container to the host ---
docker cp "$CONTAINER":"$OUT_CTR/." "$OUT_HOST/"

echo
echo "Done. Results and test conditions: $OUT_HOST"
