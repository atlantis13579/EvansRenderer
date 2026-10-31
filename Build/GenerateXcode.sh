#!/usr/bin/env bash
set -euo pipefail
build_dir="$(cd "$(dirname "$0")" && pwd)"
cmake -S "$build_dir" -B "$build_dir/Xcode" -G Xcode
