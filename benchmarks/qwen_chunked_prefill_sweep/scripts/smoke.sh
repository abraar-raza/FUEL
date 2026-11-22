#!/usr/bin/env bash
source /workspace/sweep/serve_lib.sh
D=/workspace/results/smoke_3b_512
rm -f $D/server.log
trap stop_server EXIT
start_server /workspace/models/Qwen2.5-3B-Instruct 512 $D/server.log || exit 1
run_bench /workspace/models/Qwen2.5-3B-Instruct sharegpt /workspace/datasets/sharegpt.json 30 inf 0 \
          $D smoke_3b_512_sharegpt.json $D/bench.log
echo "bench exit $?"
