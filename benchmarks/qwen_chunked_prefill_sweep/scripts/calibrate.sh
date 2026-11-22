#!/usr/bin/env bash
# Saturation calibration: request_rate=inf, 250 prompts, per (model, chunk, dataset, seed).
# One server per (model, chunk); prefix caching disabled so warmup/seed reuse can't hit cache.
source /workspace/sweep/serve_lib.sh
OUT=/workspace/results/calibration
N=${N:-250}
MODELS=(${MODELS:-Qwen2.5-3B-Instruct Qwen2.5-14B-Instruct})
CHUNKS=(${CHUNKS:-none 256 512 1024 2048 4096})
SEEDS=(${SEEDS:-0 1 2})
DATASETS=("sharegpt:/workspace/datasets/sharegpt.json" "arxiv:/workspace/datasets/arxiv_sum.jsonl")
trap stop_server EXIT
mkdir -p $OUT
for m in "${MODELS[@]}"; do
 for c in "${CHUNKS[@]}"; do
  tag=${m}_chunk-${c}; d=$OUT/$tag; mkdir -p $d
  todo=0
  for ds in "${DATASETS[@]}"; do for s in "${SEEDS[@]}"; do
    [[ -s $d/${ds%%:*}_seed${s}.json ]] || todo=1; done; done
  [[ $todo == 0 ]] && { echo "$(date +%T) skip $tag (done)"; continue; }
  echo "$(date +%T) start server $tag"
  if ! start_server /workspace/models/$m $c $d/server.log; then echo "$(date +%T) FAIL server $tag"; stop_server; continue; fi
  for ds in "${DATASETS[@]}"; do
   name=${ds%%:*}; path=${ds#*:}
   for s in "${SEEDS[@]}"; do
    f=${name}_seed${s}.json
    [[ -s $d/$f ]] && { echo "  skip $f"; continue; }
    t0=$SECONDS
    run_bench /workspace/models/$m $name $path $N inf $s $d $f $d/${name}_seed${s}.log
    rc=$?
    if [[ -s $d/$f ]]; then
      echo "$(date +%T)   $tag $f ok ($((SECONDS-t0))s) $($PY -c "import json;r=json.load(open('$d/$f'));print('rps=%.3f ok=%d E=%.0fJ'%(r['request_throughput'],r['completed'],r['energy_stats']['total_energy']))")"
    else
      echo "$(date +%T)   $tag $f FAILED rc=$rc"; tail -5 $d/${name}_seed${s}.log
      curl -sf localhost:$PORT/v1/models >/dev/null || { echo "  server gone; moving on"; break 2; }
    fi
    sleep 10
   done
  done
  stop_server
  sleep 15
 done
done
echo "$(date +%T) calibration loop finished"
