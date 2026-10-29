| dataset | cases | q_mean AR012 -> AR013 | q_min | selected | position span | KV cache | TTFT min |
|---|---:|---|---:|---:|---|---:|---:|
| deep/len32k | 4 | 1.000 -> **1.000** | 1.000 | 525 | 31077 -> **525** | 854 MiB -> **21 MiB** | 2980 ms -> 3073 ms |
| deep/multifact8k | 16 | 0.887 -> **0.975** | 0.600 | 509 | 8571 -> **515** | 238 MiB -> **21 MiB** | 3131 ms -> 3156 ms |
| deep/multifact8k-hard | 16 | 0.739 -> **0.746** | 0.643 | 513 | 8681 -> **525** | 244 MiB -> **21 MiB** | 2482 ms -> 2603 ms |
| deep/multineedle | 6 | 1.000 -> **1.000** | 1.000 | 247 | 7857 -> **265** | 217 MiB -> **14 MiB** | 1551 ms -> 1518 ms |
| explen/len16k | 4 | 1.000 -> **1.000** | 1.000 | 280 | 15576 -> **280** | 430 MiB -> **14 MiB** | 1641 ms -> 1789 ms |
| explen/len4k | 4 | 1.000 -> **1.000** | 1.000 | 146 | 3946 -> **146** | 112 MiB -> **7 MiB** | 3311 ms -> 2163 ms |
| explen/len64k | 4 | n/a -> **1.000** | 1.000 | 1163 | n/a -> **1163** | n/a -> **35 MiB** | n/a -> 6436 ms |
| explen/len8k | 4 | 1.000 -> **1.000** | 1.000 | 188 | 7812 -> **188** | 217 MiB -> **7 MiB** | 1131 ms -> 1100 ms |

`n/a` = the AR012 configuration provably cannot run that case (`context-too-long`: 63900 positions > 32768 training context).
