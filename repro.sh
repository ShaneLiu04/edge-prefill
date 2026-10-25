#!/usr/bin/env bash
# EdgePrefill one-command reproduction (AR005 FR-7).
#
#   ./repro.sh smoke    # ~<30 min on 16 cores: build + unit tests + 4K grid
#   ./repro.sh matrix   # full CPU FR-6 grid (24 points x 16K/32K + 4K/8K/mn8k)
#   ./repro.sh full     # matrix + GPU spot checks (needs CUDA + nvidia-smi)
#   ./repro.sh budget   # AR007 budget-vs-quality sweep + curve (~15 min)
#   ./repro.sh ar007    # AR007 acceptance campaign + expected gate (dense is slow)
#
# Every step is fail-fast (set -euo pipefail) and prints the exact command
# it runs. Models are downloaded with SHA256 verification into ./models
# (gitignored); datasets live in benchmarks/data and are verified in place.
# Results land in benchmarks/results/ar005-t007-matrix/ and are compared
# against benchmarks/matrix/expected.csv (non-zero exit on a violation).
#
# Options:
#   --models DIR       model cache directory (default: ./models)
#   --jobs N           build parallelism (default: nproc)
#   --skip-download    reuse an existing model cache
#   --cuda             build llama.cpp with CUDA (auto for `full` when
#                      nvidia-smi is present)
#   --no-expected      do not gate on expected.csv (exploration only)
set -euo pipefail

SUITE="${1:-smoke}"
if [[ "${SUITE}" == "--help" || "${SUITE}" == "-h" ]]; then
    sed -n '2,25p' "$0"
    exit 0
fi
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

MODELS="${ROOT}/models"
JOBS="$(nproc 2>/dev/null || echo 4)"
SKIP_DOWNLOAD=0
WANT_CUDA=0
CHECK_EXPECTED=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --cuda) WANT_CUDA=1; shift ;;
        --no-expected) CHECK_EXPECTED=0; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

step() { printf '\n=== %s\n' "$*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

MAIN_MODEL="${MODELS}/qwen2.5-1.5b-instruct-q4_k_m.gguf"
DRAFT_MODEL="${MODELS}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
EMBED_MODEL="${MODELS}/bge-small-en-v1.5-q8_0.gguf"

step "1/6 environment probe"
echo "kernel : $(uname -srm)"
echo "cpus   : ${JOBS}"
if [[ -r /proc/meminfo ]]; then
    awk '/MemTotal/ {printf "memory : %.1f GiB\n", $2/1048576}' /proc/meminfo
fi
for tool in cmake g++ python3 patch bash; do
    command -v "$tool" >/dev/null || fail "missing dependency: ${tool}"
done
cmake --version | head -1
g++ --version | head -1
python3 --version
HAS_GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
fi
if [[ "${SUITE}" == "full" && "${HAS_GPU}" == "1" ]]; then
    WANT_CUDA=1
fi

step "2/6 datasets"
DATASETS="explen/len4k explen/len8k explen/len16k deep/len32k deep/multineedle"
for ds in ${DATASETS}; do
    [[ -f "benchmarks/data/${ds}/index.tsv" && -f "benchmarks/data/${ds}/manifest.json" ]] \
        || fail "dataset missing: benchmarks/data/${ds} (regenerate with benchmarks/scripts/gen_niah.py / gen_multineedle.py)"
done
echo "datasets present: ${DATASETS}"

step "3/6 models"
if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
    [[ -f "${MAIN_MODEL}" ]] || fail "--skip-download but ${MAIN_MODEL} is absent"
    echo "reusing ${MODELS}"
else
    bash scripts/download_model.sh --all "${MODELS}"
fi
[[ -f "${EMBED_MODEL}" ]] || fail "embedding model missing: ${EMBED_MODEL}"

step "4/6 build"
BUILD_DIR="${ROOT}/build"
CUDA_FLAG="OFF"
if [[ "${WANT_CUDA}" == "1" ]]; then
    CUDA_FLAG="ON"
fi
echo "cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DEDGE_PREFILL_WITH_CUDA=${CUDA_FLAG}"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
      -DEDGE_PREFILL_WITH_CUDA="${CUDA_FLAG}" >/dev/null
cmake --build "${BUILD_DIR}" -j "${JOBS}"
BIN="${BUILD_DIR}/edge-prefill"

step "5/6 unit + integration tests"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

step "6/6 benchmark suite: ${SUITE}"
OUT="${ROOT}/benchmarks/results/ar005-t007-matrix"
MATRIX_ARGS=(
    --bin "${BIN}"
    --model "${MAIN_MODEL}"
    --embedding "${EMBED_MODEL}"
    --data-root "${ROOT}/benchmarks/data"
    --out "${OUT}"
)
case "${SUITE}" in
    smoke)
        SMOKE_OUT="${ROOT}/benchmarks/results/ar005-smoke"
        python3 benchmarks/scripts/run_matrix_ar005.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --embedding "${EMBED_MODEL}" \
            --data-root "${ROOT}/benchmarks/data" \
            --out "${SMOKE_OUT}" --smoke --repeats 1 --dense-repeats 1
        python3 benchmarks/scripts/run_matrix_ar005.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --embedding "${EMBED_MODEL}" \
            --data-root "${ROOT}/benchmarks/data" \
            --out "${SMOKE_OUT}" --assemble-only
        OUT="${SMOKE_OUT}"
        ;;
    matrix)
        python3 benchmarks/scripts/run_matrix_ar005.py "${MATRIX_ARGS[@]}" \
            --repeats 3 --dense-repeats 1
        python3 benchmarks/scripts/run_matrix_ar005.py "${MATRIX_ARGS[@]}" --assemble-only
        ;;
    full)
        python3 benchmarks/scripts/run_matrix_ar005.py "${MATRIX_ARGS[@]}" \
            --repeats 3 --dense-repeats 1
        python3 benchmarks/scripts/run_matrix_ar005.py "${MATRIX_ARGS[@]}" --assemble-only
        if [[ "${HAS_GPU}" == "1" ]]; then
            GPU_OUT="${ROOT}/benchmarks/results/ar005-t007-gpu"
            python3 benchmarks/scripts/run_matrix_ar005.py \
                --bin "${BIN}" --model "${MAIN_MODEL}" \
                --embedding "${EMBED_MODEL}" \
                --data-root "${ROOT}/benchmarks/data" \
                --out "${GPU_OUT}" \
                --datasets len16k,len32k \
                --confs b025-a24-pool1-qgv2.conf,b025-a24-pool0-qgoff.conf \
                --repeats 3 --dense-repeats 1
            python3 benchmarks/scripts/run_matrix_ar005.py \
                --bin "${BIN}" --model "${MAIN_MODEL}" \
                --embedding "${EMBED_MODEL}" \
                --data-root "${ROOT}/benchmarks/data" \
                --out "${GPU_OUT}" --assemble-only
        else
            echo "no GPU detected: full suite runs the CPU matrix only"
        fi
        ;;
    budget)
        # AR007 FR-5: budget-vs-quality sweep + curve on the 4K/8K cells.
        # Produces benchmarks/results/ar007-sweep/{summary.md,budget_curve.txt}.
        python3 benchmarks/scripts/sweep_budget.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --embedding "${EMBED_MODEL}" \
            --datasets explen/len4k,explen/len8k \
            --budgets 0.05,0.08,0.12,0.20 \
            --signals lexical,embedding \
            --repeats 1 \
            --out "${ROOT}/benchmarks/results/ar007-sweep"
        OUT="${ROOT}/benchmarks/results/ar007-sweep"
        ;;
    ar007)
        # AR007 FR-5: full acceptance campaign (dense / curve / AR006 budget)
        # plus the machine-independent gate. Dense dominates the runtime
        # (32K is minutes per case); resumes by skipping finished cells.
        bash benchmarks/scripts/ar007_accept.sh \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --out "${ROOT}/benchmarks/results/ar007-campaign"
        python3 benchmarks/scripts/check_expected_ar007.py \
            "${ROOT}/benchmarks/results/ar007-campaign" \
            "${ROOT}/benchmarks/matrix/budget/expected.csv" \
            || fail "AR007 campaign deviates from benchmarks/matrix/budget/expected.csv"
        OUT="${ROOT}/benchmarks/results/ar007-campaign"
        ;;
    *)
        fail "unknown suite: ${SUITE} (smoke|matrix|full|budget|ar007)"
        ;;
esac

if [[ -f "${OUT}/matrix.md" ]]; then
    echo
    cat "${OUT}/matrix.md"
fi

if [[ "${CHECK_EXPECTED}" == "1" && "${SUITE}" != "smoke" && "${SUITE}" != "budget" \
      && "${SUITE}" != "ar007" ]]; then
    step "expected-results gate"
    python3 benchmarks/scripts/check_expected.py \
        --matrix "${OUT}/matrix.csv" \
        --expected "${ROOT}/benchmarks/matrix/expected.csv" \
        --datasets len16k,len32k,len4k,len8k,mn8k \
        || fail "matrix deviates from benchmarks/matrix/expected.csv (see floors above)"
fi

step "done"
echo "raw CSVs: ${OUT}/*/*_raw.csv"
echo "summary : ${OUT}/matrix.csv / matrix.md"
