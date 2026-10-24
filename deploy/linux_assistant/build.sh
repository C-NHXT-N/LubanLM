#!/bin/sh
set -eu
cd "$(dirname "$0")"
test -s include/rkllm.h
test -s lib/librkllmrt.so
python3 frontend.py --root . --verify-only
g++ -O2 -std=c++17 -Wall -Wextra -Iinclude board_chat.cpp -Llib -lrkllmrt -Wl,-rpath,'$ORIGIN/lib' -o board_chat
python3 -c 'import hashlib,json,pathlib; p=pathlib.Path("board_chat"); pathlib.Path("build-manifest.json").write_text(json.dumps({"board_chat_sha256":hashlib.sha256(p.read_bytes()).hexdigest()}))'
