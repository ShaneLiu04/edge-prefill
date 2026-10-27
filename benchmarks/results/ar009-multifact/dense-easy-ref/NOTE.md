# Dense reference for the AR009 easy tier (not a candidate)

`budget = 1.0` on a **4-row subset** of `deep/multifact8k` (cases mf-01, mf-02).
A full 16-row dense pass exceeds the session time budget, so the ceiling was
measured on a subset instead of inferred. The subset was built by filtering the
canonical `index.tsv` (`subset_index.tsv`) with `contexts/` symlinked to the
canonical directory; no context text was altered.

Exact command:

```bash
./build-wsl/edge-prefill bench --config bench.conf \
  --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --data-dir <subset dir> --out <this dir>
```

Result: q = 1.000 / 1.000 / 1.000 / 0.900, mean 0.975 (see `*_raw.csv`).
