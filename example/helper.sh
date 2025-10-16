#!/bin/bash

# simple dispatcher: first arg is the command (pred|eval), the rest are forwarded
PYTHON=python3
MODEL="Meta-Llama-3.1-8B-Instruct"
LONG_BENCH_DS_PATH="/home/ysy/code/research/longbench_data"
WORLD_SIZE=1
# DATASET="evol-instruct-python-1k"
DATASET="2wikimqa"

# if use TuneCache
USE_TUNECACHE=1
SENDER_MODEL="Meta-Llama-3.1-8B-Instruct"


cmd="$1"
shift || true

# detect --debug flag anywhere in the remaining args; remove it and enable DEBUG
DEBUG_ARG=""
new_args=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--debug" ]]; then
    DEBUG_ARG="-m debugpy --listen 5678 --wait-for-client"
    shift
    continue
  fi
  new_args+=("$1")
  shift
done
# restore positional parameters to cleaned args
set -- "${new_args[@]}"

usage() {
  echo "Usage: $0 {pred|eval|dump} [other args...]"
  echo "Commands:"
  echo "  pred    Run prediction"
  echo "  eval    Run evaluation"
  echo "  dump    Dump query/key/value cache"
}

if [[ -z "$cmd" ]]; then
  usage
fi

if [[ "$USE_TUNECACHE" -eq 1 ]]; then
  echo "Using TuneCache with sender model: $SENDER_MODEL, receiver model: $MODEL"
  TUNE_CACHE_ARGS="--use_tune_cache --sender_model $SENDER_MODEL"
fi

case "$cmd" in
  pred)
      "$PYTHON" \
        $DEBUG_ARG \
        tunecache_pred.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --long_bench_ds_path "$LONG_BENCH_DS_PATH" \
        $TUNE_CACHE_ARGS \
        "$@"
    ;;
  eval)
      "$PYTHON" \
        $DEBUG_ARG \
        tunecache_pred.py \
        --model "$MODEL" \
        "$@"
    ;;
  dump)
      "$PYTHON" \
        $DEBUG_ARG \
        tunecache_pred.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --long_bench_ds_path "$LONG_BENCH_DS_PATH" \
        --dump_qkv \
        $TUNE_CACHE_ARGS \
        "$@"
    ;;
  "")
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    ;;
esac



