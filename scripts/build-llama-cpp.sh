#!/usr/bin/env bash
# Build the locked llama.cpp for every backend this host has (ADR 0009).
#
#   scripts/build-llama-cpp.sh              # auto: HIP and/or CUDA if present, else CPU
#   scripts/build-llama-cpp.sh cuda         # one backend
#   scripts/build-llama-cpp.sh hip vulkan   # several
#
# Each backend builds into its own directory under upstream/llama.cpp (see
# build.backends in the lock); HIP keeps "build" so existing paths still work.
# GPU targets are detected by scripts/detect_backends.py; HIP_GPU_TARGETS and
# CUDA_ARCHITECTURES override detection when building for a card not installed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/upstream/llama-cpp.lock.json"
SRC="$ROOT/upstream/llama.cpp"
JOBS="$(nproc)"

# Capture first: readarray with process substitution hides a failing producer.
META_TEXT=$(python3 -c '
import json,sys
p=json.load(open(sys.argv[1]))
print(p["repository"])
print(p["branch"])
print(p["commit"])
' "$LOCK")
readarray -t META <<< "$META_TEXT"
REPOSITORY="${META[0]}"
BRANCH="${META[1]}"
EXPECTED="${META[2]}"

[[ -f "$SRC/.git" ]] || {
    echo "missing llama.cpp submodule: git submodule update --init --recursive upstream/llama.cpp" >&2
    exit 2
}
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$EXPECTED" ]] || {
    echo "llama.cpp HEAD does not match locked $BRANCH ($EXPECTED)" >&2
    exit 2
}
# Preserve fail-fast status collection. The verifier accepts only the exact
# SHA256-declared patch tree, not arbitrary dirty source or a missing patch.
SOURCE_STATUS=$(git -C "$SRC" status --porcelain --untracked-files=no)
python3 "$ROOT/scripts/verify_llama_patches.py" --root "$ROOT" --source "$SRC"
[[ "$REPOSITORY" == "https://github.com/ggml-org/llama.cpp.git" ]] || {
    echo "unexpected llama.cpp repository in lock" >&2
    exit 2
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
    echo "nproc did not return a positive integer" >&2
    exit 2
}

# Use one device snapshot for the entire build, rather than querying every
# field independently (which could observe different devices/driver states).
DETECTED=$(python3 "$ROOT/scripts/detect_backends.py")
field() {  # field <backend.field>
    python3 -c '
import json,sys
backend, name = sys.argv[1].split(".")
value = json.load(sys.stdin)[backend][name]
print(";".join(value) if isinstance(value, list) else ("" if value is None else value))
' "$1" <<< "$DETECTED"
}
build_dir() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["build"]["backends"][sys.argv[2]]["build_dir"])' "$LOCK" "$1"
}

if [[ $# -eq 0 || "$*" == auto ]]; then
    AUTO_TEXT=$(python3 -c '
import json,sys
p=json.load(sys.stdin)
chosen=[]
if p["hip"]["targets"]: chosen.append("hip")
if p["cuda"]["architectures"]: chosen.append("cuda")
print("\n".join(chosen or ["cpu"]))
' <<< "$DETECTED")
    readarray -t BACKENDS <<< "$AUTO_TEXT"
else
    BACKENDS=("$@")
fi

COMMON=(
    -G Ninja
    -DGGML_NATIVE=ON
    -DGGML_LTO=ON
    -DGGML_CPU_REPACK=ON
    -DGGML_HIP=OFF
    -DGGML_CUDA=OFF
    -DGGML_VULKAN=OFF
    -DCMAKE_BUILD_TYPE=Release
)

configure() {  # configure <backend> <build dir>
    local backend=$1 build=$2
    case $backend in
        hip)
            GPU_TARGETS="$(field hip.targets)"
            [[ -n "$GPU_TARGETS" ]] || {
                echo "hip: no ROCm GPU in the KFD topology; set HIP_GPU_TARGETS to build for one anyway" >&2
                exit 2
            }
            command -v hipconfig >/dev/null || {
                echo "hip: GPU targets $GPU_TARGETS found but no ROCm toolchain (hipconfig); install ROCm/HIP" >&2
                exit 2
            }
            # Exported, not prefixed onto the configure line: the HIP clang resolves
            # its own install root while compiling, and ROCm 7.2's rocm-llvm guesses
            # /opt/rocm/lib from its binary path, missing hip/hip_fp16.h. A login
            # shell masks this via /etc/profile.d/rocm.sh (ROCM_PATH); a shell older
            # than the ROCm install does not.
            HIP_PATH=$(hipconfig -R)
            export HIP_PATH
            HIP_COMPILER_DIR=$(hipconfig -l)
            HIPCXX="$HIP_COMPILER_DIR/clang" \
            cmake -S "$SRC" -B "$build" "${COMMON[@]}" \
                -DGGML_HIP=ON \
                -DGGML_HIP_GRAPHS=ON \
                -DGGML_CUDA_FA=ON \
                -DGPU_TARGETS="$GPU_TARGETS"
            ;;
        cuda)
            CUDA_ARCHS="$(field cuda.architectures)"
            [[ -n "$CUDA_ARCHS" ]] || {
                echo "cuda: no NVIDIA GPU detected ($(field cuda.reason)); set CUDA_ARCHITECTURES to build for one anyway" >&2
                exit 2
            }
            NVCC="$(field cuda.nvcc)"
            [[ -n "$NVCC" ]] || {
                echo "cuda: NVIDIA architectures $CUDA_ARCHS found but no nvcc; install the CUDA toolkit" >&2
                exit 2
            }
            cmake -S "$SRC" -B "$build" "${COMMON[@]}" \
                -DGGML_CUDA=ON \
                -DGGML_CUDA_FA=ON \
                -DCMAKE_CUDA_COMPILER="$NVCC" \
                -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
            ;;
        vulkan)
            command -v glslc >/dev/null || {
                echo "vulkan: no glslc; install shaderc and the Vulkan headers" >&2
                exit 2
            }
            cmake -S "$SRC" -B "$build" "${COMMON[@]}" -DGGML_VULKAN=ON
            ;;
        cpu)
            cmake -S "$SRC" -B "$build" "${COMMON[@]}"
            ;;
        *)
            echo "unknown backend: $backend (expected auto, hip, cuda, vulkan or cpu)" >&2
            exit 2
            ;;
    esac
}

for backend in "${BACKENDS[@]}"; do
    [[ $backend =~ ^(hip|cuda|vulkan|cpu)$ ]] || {
        echo "unknown backend: $backend (expected auto, hip, cuda, vulkan or cpu)" >&2
        exit 2
    }
done
for backend in "${BACKENDS[@]}"; do
    BUILD="$SRC/$(build_dir "$backend")"
    echo "== llama.cpp $backend -> $BUILD"
    configure "$backend" "$BUILD"
    cmake --build "$BUILD" --parallel "$JOBS" --target \
        llama-server llama-cli llama-quantize llama-gguf
    for binary in llama-server llama-cli llama-quantize llama-gguf; do
        test -x "$BUILD/bin/$binary"
    done
    "$BUILD/bin/llama-server" --version
    printf 'llama.cpp %s (%s) %s built at %s\n' "$BRANCH" "$EXPECTED" "$backend" "$BUILD"
done
