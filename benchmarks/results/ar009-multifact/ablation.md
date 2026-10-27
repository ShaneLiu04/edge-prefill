# AR009 multi-fact coverage ablation

model_sha256 = 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e

budget = 0.05, phrase_weight = 0.5, max_new_tokens = 48, threads = 16

| dataset | coverage_weight | cases | q_mean | q_min | covered facts | selected tok | TTFT ms |
|---|---|---|---|---|---|---|---|
| deep/multifact8k | 0 | 16 | 0.488 | 0.429 | 0.00 | 443 | 2767 |
| deep/multifact8k | 3 | 16 | 0.878 | 0.706 | 3.25 | 443 | 2927 |
| deep/multifact8k-hard | 0 | 16 | 0.261 | 0.261 | 0.00 | 445 | 2642 |
| deep/multifact8k-hard | 3 | 16 | 0.652 | 0.429 | 6.50 | 445 | 3218 |

| dataset | q(AR008) | q(AR009) | delta |
|---|---|---|---|
| deep/multifact8k | 0.488 | 0.878 | +0.390 |
| deep/multifact8k-hard | 0.261 | 0.652 | +0.391 |
