#!/usr/bin/env bash
export EDGE_PREFILL_TEST_MODEL=/mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
BIN=/mnt/d/prefill/build-wsl/edge_prefill_tests
for t in classify_threshold_extremes classify_probe_exceeds_n_ctx_fails classify_probe_exactly_n_ctx_runs classify_invalid_params_rejected probe_argmax; do
    echo "=== $t ==="
    "$BIN" "$t" 2>&1 | grep -E '^PASS|^FAIL|ASSERT|failure' | head -4
done
