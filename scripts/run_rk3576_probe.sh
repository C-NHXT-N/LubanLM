#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RKLLM_LOG_LEVEL=1
name="${1:?model name required}"
case "$name" in Qwen3.5-2B|Qwen3-4B-Instruct-2507) ;; *) exit 2;; esac
model="models/$name/${name}_w4a16_rk3576_ctx2048.rkllm"
test -s "$model"
mkdir -p results
exec ./board_probe "$model" "models/$name/native-tokens.tsv" "results/$name.jsonl" 256 2048
