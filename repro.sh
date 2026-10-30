#!/usr/bin/env bash
# EdgePrefill one-command reproduction (AR005 FR-7).
#
#   ./repro.sh smoke    # ~<30 min on 16 cores: build + unit tests + 4K grid
#   ./repro.sh matrix   # full CPU FR-6 grid (24 points x 16K/32K + 4K/8K/mn8k)
#   ./repro.sh full     # matrix + GPU spot checks (needs CUDA + nvidia-smi)
#   ./repro.sh budget   # AR007 budget-vs-quality sweep + curve (~15 min)
#   ./repro.sh ar007    # AR007 acceptance campaign + expected gate (dense is slow)
#   ./repro.sh ar008    # AR008 combination matrix + portfolio gate (dense is slow)
#   ./repro.sh ar009    # AR009 multi-fact coverage ablation + expected gate
#   ./repro.sh ar010    # AR010 evidence-first admission: matrix + table + gate
#   ./repro.sh ar011    # AR011 structure-bounded admission: 7-set matrix + gate
#   ./repro.sh ar012    # AR012 sentence-complete admission: matrix + facts + gate
#   ./repro.sh ar013    # AR013 compact context: matrix + KV/latency/capability gate
#   ./repro.sh ar014    # AR014 session KV reuse: multiturn two arms + gate + injections
#   ./repro.sh ar015    # AR015 KV snapshots: cold/warm 8 datasets + restart resume + gate
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
#   --llama-dir DIR    use an existing llama.cpp tree (also: EDGE_PREFILL_LLAMA_DIR)
#                      - needed behind a proxy or on an air-gapped server
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
# Offline / air-gapped builds: point at an existing llama.cpp tree instead of
# letting CMake fetch the pinned tarball (behind a proxy the download can 407).
LLAMA_DIR="${EDGE_PREFILL_LLAMA_DIR:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --cuda) WANT_CUDA=1; shift ;;
        --no-expected) CHECK_EXPECTED=0; shift ;;
        --llama-dir) LLAMA_DIR="$2"; shift 2 ;;
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
CMAKE_EXTRA=()
if [[ -n "${LLAMA_DIR}" ]]; then
    CMAKE_EXTRA+=("-DEDGE_PREFILL_LLAMA_DIR=${LLAMA_DIR}")
    echo "using existing llama.cpp tree: ${LLAMA_DIR}"
fi
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
      -DEDGE_PREFILL_WITH_CUDA="${CUDA_FLAG}" ${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"} >/dev/null
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
    ar008)
        # AR008 FR-4: measure the execution-combination matrix, derive the
        # portfolio table from it, then gate the routed combination against
        # the best fixed one (Pareto, machine independent).
        python3 benchmarks/scripts/portfolio_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --out "${ROOT}/benchmarks/results/ar008-portfolio"
        python3 benchmarks/scripts/check_portfolio.py \
            "${ROOT}/benchmarks/results/ar008-portfolio/matrix.csv" \
            "${ROOT}/benchmarks/results/ar008-portfolio/portfolio.json" \
            || fail "routed combination is worse than the best fixed combination"
        OUT="${ROOT}/benchmarks/results/ar008-portfolio"
        ;;
    ar010)
        # AR010 FR-4: measure the five-arm matrix, derive the routing table from
        # it and gate the routed result (Pareto + injected degraded arm).
        python3 benchmarks/scripts/ar010_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --out "${ROOT}/benchmarks/results/ar010-evidence-first"
        python3 benchmarks/scripts/ar010_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --out "${ROOT}/benchmarks/results/ar010-evidence-first" \
            --datasets deep/multifact8k,deep/multifact8k-hard \
            --arms selection,selection-prior,gate-p16,gate-p24-local32 \
            --flat-budget 0.05 --max-new-tokens 48
        python3 benchmarks/scripts/ar010_report.py \
            "${ROOT}/benchmarks/results/ar010-evidence-first"
        python3 benchmarks/scripts/ar010_report.py \
            "${ROOT}/benchmarks/results/ar010-evidence-first" --gate \
            || fail "routed arm is worse than the best fixed arm"
        python3 benchmarks/scripts/ar010_report.py \
            "${ROOT}/benchmarks/results/ar010-evidence-first" --gate \
            --inject deep/len32k:selection-prior > /dev/null 2>&1 \
            && fail "gate did not reject the injected degraded arm"
        OUT="${ROOT}/benchmarks/results/ar010-evidence-first"
        ;;
    ar011)
        # AR011 FR-4: measure the seven-dataset matrix against AR010, derive the
        # no-regression table and gate it (quality, tokens, TTFT) including an
        # injected degradation to prove the gate can fail. Sparse arms only:
        # the multi-fact families need a 48-token answer and a flat 0.05 budget,
        # which the matrix script applies per dataset.
        python3 benchmarks/scripts/ar011_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --out "${ROOT}/benchmarks/results/ar011-structure-bounded"
        python3 benchmarks/scripts/ar011_report.py \
            "${ROOT}/benchmarks/results/ar011-structure-bounded"
        python3 benchmarks/scripts/ar011_report.py \
            "${ROOT}/benchmarks/results/ar011-structure-bounded" --gate \
            || fail "AR011 routing is worse than AR010 (quality, tokens or TTFT)"
        python3 benchmarks/scripts/ar011_report.py \
            "${ROOT}/benchmarks/results/ar011-structure-bounded" --gate \
            --inject explen/len8k:ar010 > /dev/null 2>&1 \
            && fail "gate did not reject the injected degraded arm"
        OUT="${ROOT}/benchmarks/results/ar011-structure-bounded"
        ;;
    ar012)
        # AR012: sentence-complete admission. Measures the 4-arm matrix (with the
        # admitted text dumped for the deterministic fact audit), derives the
        # routing table, gates it against AR011 (no free cost) and checks that
        # every asked fact value is present in the admitted context.
        python3 benchmarks/scripts/ar012_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" --dump \
            --out "${ROOT}/benchmarks/results/ar012-sentence-complete"
        python3 benchmarks/scripts/ar012_report.py \
            "${ROOT}/benchmarks/results/ar012-sentence-complete"
        python3 benchmarks/scripts/ar012_report.py \
            "${ROOT}/benchmarks/results/ar012-sentence-complete" --gate \
            || fail "AR012 routing regresses against AR011"
        python3 benchmarks/scripts/ar012_report.py \
            "${ROOT}/benchmarks/results/ar012-sentence-complete" --gate \
            --inject deep/multifact8k:ar012-trim > /dev/null 2>&1 \
            && fail "gate did not reject the injected degraded arm"
        python3 benchmarks/scripts/check_facts_admission.py \
            "${ROOT}/benchmarks/results/ar012-sentence-complete" \
            --dataset deep/multifact8k-hard --floor 0.95 --gate-arm ar012 \
            || fail "asked facts are missing from the admitted context"
        OUT="${ROOT}/benchmarks/results/ar012-sentence-complete"
        ;;
    ar013)
        # AR013: compact context. Measures the 8-dataset matrix (7 existing plus a
        # 64K document that the AR012 configuration provably cannot run), gates
        # quality / latency / KV allocation / long-context capability, and proves
        # the gate can fail with two injected degradations.
        python3 benchmarks/scripts/gen_niah.py \
            --out "${ROOT}/benchmarks/data/explen/len64k" --cases 4 --words 57000
        python3 benchmarks/scripts/ar013_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" --dump \
            --out "${ROOT}/benchmarks/results/ar013-compact-context"
        python3 benchmarks/scripts/ar013_report.py \
            "${ROOT}/benchmarks/results/ar013-compact-context"
        python3 benchmarks/scripts/ar013_report.py \
            "${ROOT}/benchmarks/results/ar013-compact-context" --gate \
            || fail "AR013 regresses on quality, latency, KV or 64K capability"
        python3 benchmarks/scripts/ar013_report.py \
            "${ROOT}/benchmarks/results/ar013-compact-context" --gate \
            --inject explen/len4k:no-compaction > /dev/null 2>&1 \
            && fail "gate did not reject the injected no-compaction arm"
        python3 benchmarks/scripts/ar013_report.py \
            "${ROOT}/benchmarks/results/ar013-compact-context" --gate \
            --inject explen/len64k:no-capability > /dev/null 2>&1 \
            && fail "gate did not reject the injected lost-capability arm"
        OUT="${ROOT}/benchmarks/results/ar013-compact-context"
        ;;
    ar014)
        # AR014: session KV reuse. Regenerates the multi-turn corpus, runs the two
        # arms (recompute the whole log vs reuse the prefix KV) plus single-turn
        # spot checks, gates quality equivalence / per-turn latency / audit, and
        # proves the gate can fail with two injected degradations.
        python3 benchmarks/scripts/gen_multiturn.py \
            --out "${ROOT}/benchmarks/data/deep/multiturn"
        python3 benchmarks/scripts/ar014_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --datasets deep/multiturn,explen/len4k,deep/multifact8k --force
        python3 benchmarks/scripts/ar014_report.py --gate \
            || fail "AR014 regresses on quality equivalence or per-turn latency"
        python3 benchmarks/scripts/ar014_report.py --gate \
            --inject deep/multiturn:no-reuse > /dev/null 2>&1 \
            && fail "gate did not reject the injected no-reuse arm"
        python3 benchmarks/scripts/ar014_report.py --gate \
            --inject deep/multiturn:quality-drift > /dev/null 2>&1 \
            && fail "gate did not reject the injected answer drift"
        OUT="${ROOT}/benchmarks/results/ar014-session-kv-reuse"
        ;;
    ar015)
        # AR015: persistent KV snapshots. Cold and warm arms run as separate
        # processes over one snapshot directory, plus a conversation replayed
        # after a "restart"; the gate checks hit rate, bit-identical quality,
        # warm/cold latency and snapshot size, and two injections must fail.
        python3 benchmarks/scripts/ar015_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --datasets deep/multiturn,explen/len4k,explen/len64k --force
        python3 benchmarks/scripts/ar015_report.py --gate \
            || fail "AR015 snapshots are not faster or not bit-identical"
        python3 benchmarks/scripts/ar015_report.py --gate \
            --inject explen/len4k:no-hit > /dev/null 2>&1 \
            && fail "gate did not reject the injected no-hit arm"
        python3 benchmarks/scripts/ar015_report.py --gate \
            --inject explen/len4k:quality-drift > /dev/null 2>&1 \
            && fail "gate did not reject the injected answer drift"
        OUT="${ROOT}/benchmarks/results/ar015-kv-snapshot"
        ;;
    ar016)
        # AR016: execution plan + latency-targeted budget. The cost model comes
        # from a separate `probe` run (never from the data being optimised), the
        # matrix compares the hardware-default plan with the probed one and with
        # the SLA planner, and the gate checks placement/threads/quality/speedup.
        rm -rf "${ROOT}/benchmarks/results/ar016-exec-plan/probe-plan"
        "${BIN}" probe --model "${MAIN_MODEL}" \
            --prompt-file "${ROOT}/benchmarks/data/explen/len4k/contexts/niah-01.txt" \
            --threads 12 --out "${ROOT}/benchmarks/results/ar016-exec-plan/probe-plan" \
            || fail "AR016 probe could not fit the prefill cost model"
        python3 benchmarks/scripts/ar016_matrix.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --plan-dir benchmarks/results/ar016-exec-plan/probe-plan \
            --base-threads 16 --sla-target-ms 2500 --repeats 2 --warmup 1 \
            --datasets explen/len4k,explen/len8k,deep/multifact8k,deep/multifact8k-hard \
            --force --out benchmarks/results/ar016-exec-plan \
            || fail "AR016 matrix failed"
        python3 benchmarks/scripts/ar016_report.py \
            benchmarks/results/ar016-exec-plan --gate --sla-target-ms 2500 \
            || fail "AR016 plan is not honoured or is not faster"
        python3 benchmarks/scripts/ar016_report.py \
            benchmarks/results/ar016-exec-plan --gate --inject cost-skew \
            --sla-target-ms 2500 > /dev/null 2>&1 \
            && fail "gate did not reject the injected cost-model skew"
        OUT="${ROOT}/benchmarks/results/ar016-exec-plan"
        ;;
    ar009)
        # AR009 FR-3: regenerate the deterministic multi-fact families, run the
        # coverage ablation (AR008 ranking vs coverage-fair fill) and gate the
        # result. Both tiers are selection-level only, so this is fast.
        python3 benchmarks/scripts/gen_multifact.py \
            --out "${ROOT}/benchmarks/data/deep/multifact8k" \
            --words 7500 --cases 8 --facts 6 --asked 3 --rows-per-context 2 --seed 41
        python3 benchmarks/scripts/gen_multifact.py \
            --out "${ROOT}/benchmarks/data/deep/multifact8k-hard" \
            --words 7500 --cases 8 --facts 12 --asked 6 --rows-per-context 2 --seed 43
        python3 benchmarks/scripts/ar009_ablation.py \
            --bin "${BIN}" --model "${MAIN_MODEL}" \
            --out "${ROOT}/benchmarks/results/ar009-multifact" --force
        python3 benchmarks/scripts/check_expected_ar009.py \
            "${ROOT}/benchmarks/results/ar009-multifact" \
            "${ROOT}/benchmarks/matrix/multifact/expected_ar009.csv" \
            || fail "multi-fact coverage is below the AR009 floors"
        OUT="${ROOT}/benchmarks/results/ar009-multifact"
        ;;
    *)
        fail "unknown suite: ${SUITE} (smoke|matrix|full|budget|ar007..ar016)"
        ;;
esac

if [[ -f "${OUT}/matrix.md" ]]; then
    echo
    cat "${OUT}/matrix.md"
fi
if [[ -f "${OUT}/ablation.md" ]]; then
    echo
    cat "${OUT}/ablation.md"
fi

# The AR campaigns carry their own gates (floors + Pareto/no-free-cost), so the
# generic expected.csv check only applies to the matrix-style suites.
if [[ "${CHECK_EXPECTED}" == "1" && "${SUITE}" != "smoke" && "${SUITE}" != "budget" \
      && "${SUITE}" != ar0* ]]; then
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
