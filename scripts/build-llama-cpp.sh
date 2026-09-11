#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/upstream/llama-cpp.lock.json"
SRC="$ROOT/upstream/llama.cpp"
BUILD="$SRC/build"
JOBS="$(nproc)"

readarray -t META < <(python3 -c '
import json,sys
p=json.load(open(sys.argv[1]))
print(p["repository"])
print(p["branch"])
print(p["commit"])
print(";".join(p["build"]["gpu_targets"]))
' "$LOCK")
REPOSITORY="${META[0]}"
BRANCH="${META[1]}"
EXPECTED="${META[2]}"
GPU_TARGETS="${META[3]}"

[[ -f "$SRC/.git" ]] || {
    echo "missing llama.cpp submodule: git submodule update --init --recursive upstream/llama.cpp" >&2
    exit 2
}
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$EXPECTED" ]] || {
    echo "llama.cpp HEAD does not match locked $BRANCH ($EXPECTED)" >&2
    exit 2
}
[[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]] || {
    echo "llama.cpp tracked worktree is dirty" >&2
    exit 2
}
[[ "$REPOSITORY" == "https://github.com/ggml-org/llama.cpp.git" ]] || {
    echo "unexpected llama.cpp repository in lock" >&2
    exit 2
}
[[ "$GPU_TARGETS" == "gfx1030" ]] || {
    echo "unexpected GPU target in lock: $GPU_TARGETS" >&2
    exit 2
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
    echo "nproc did not return a positive integer" >&2
    exit 2
}

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DGGML_HIP=ON \
    -DGGML_NATIVE=ON \
    -DGGML_LTO=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CPU_REPACK=ON \
    -DGPU_TARGETS="$GPU_TARGETS" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD" --parallel "$JOBS" --target \
    llama-server llama-cli llama-quantize llama-gguf

for binary in llama-server llama-cli llama-quantize llama-gguf; do
    test -x "$BUILD/bin/$binary"
done

"$BUILD/bin/llama-server" --version
printf 'llama.cpp %s (%s) built at %s\n' "$BRANCH" "$EXPECTED" "$BUILD"
