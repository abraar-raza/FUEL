#!/usr/bin/env bash
# Shared helpers. Always run from /workspace so venv imports site-packages vllm 0.34.0.
PY=/workspace/venv/bin/python
BENCH=/workspace/repo/benchmarks/benchmark_serving.py
PORT=${PORT:-8000}

# start_server MODEL_PATH CHUNK LOGFILE   (CHUNK = none | <int>)
start_server() {
  local model=$1 chunk=$2 log=$3
  local cargs
  if [[ $chunk == none ]]; then
    cargs=(--no-enable-chunked-prefill)          # max_num_batched_tokens -> max_model_len (32768)
  else
    cargs=(--enable-chunked-prefill --max-num-batched-tokens "$chunk")
  fi
  cd /workspace
  CUDA_VISIBLE_DEVICES=0 nohup $PY -m vllm.entrypoints.openai.api_server \
      --model "$model" --host 127.0.0.1 --port $PORT \
      --no-enable-prefix-caching "${cargs[@]}" > "$log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 900); do
    if curl -sf localhost:$PORT/v1/models >/dev/null; then return 0; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died, see $log"; return 1; fi
    sleep 1
  done
  echo "server start timeout"; return 1
}

stop_server() {
  [[ -n $SERVER_PID ]] && kill $SERVER_PID 2>/dev/null
  for i in $(seq 1 60); do kill -0 $SERVER_PID 2>/dev/null || break; sleep 1; done
  kill -9 $SERVER_PID 2>/dev/null
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  # wait for GPU memory to be released
  for i in $(seq 1 60); do
    [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) -lt 1000 ]] && break; sleep 1
  done
  SERVER_PID=
}

# run_bench MODEL DATASET_NAME DATASET_PATH NUM_PROMPTS RATE SEED RESULT_DIR RESULT_FILE LOG [EXTRA ARGS...]
run_bench() {
  cd /workspace
  $PY $BENCH --backend vllm --host 127.0.0.1 --port $PORT --endpoint /v1/completions \
     --model "$1" --dataset-name "$2" --dataset-path "$3" --num-prompts "$4" \
     --request-rate "$5" --seed "$6" --save-result --result-dir "$7" --result-filename "$8" \
     --gpu-ids 0 --gpu-monitor-interval 0.025 --disable-tqdm "${@:10}" > "$9" 2>&1
}
