#!/bin/bash

# simple dispatcher: first arg is the command (pred|eval), the rest are forwarded
PYTHON=python3
MODEL="Meta-Llama-3-8B-Instruct"
LONG_BENCH_DS_PATH="/home/ysy/code/research/longbench_data"
WORLD_SIZE=1
DATASET="2wikimqa"

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
  echo "Usage: $0 {pred|eval} [--model MODEL] [other args...]"
  echo "Examples:"
  echo "  $0 pred --model Meta-Llama-3.1-8B-Instruct --dataset 2wikimqa"
  echo "  $0 eval --model meta-llama-3.1-8b-instruct"
}

if [[ -z "$cmd" ]]; then
  usage
fi

case "$cmd" in
  pred)
      "$PYTHON" \
        $DEBUG_ARG \
        pred.py \
        --model "$MODEL" \
        --world_size "$WORLD_SIZE" \
        --dataset "$DATASET" \
        --long_bench_ds_path "$LONG_BENCH_DS_PATH" \
        "$@"
    ;;
  eval)
      "$PYTHON" \
        $DEBUG_ARG \
        eval.py \
        --model "$MODEL" \
        "$@"
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    ;;
esac



