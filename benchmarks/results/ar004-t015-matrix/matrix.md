# AR004 T015 main experiment matrix

| length | config | device | speedup | quality(med) | source |
|---|---|---|---:|---:|---|
| 16k | attn | CPU | 3.92x | 1.000 | ar004-t002-final/len16k-attn |
| 16k | attn | GPU | 14.42x | 1.000 | ar004-t012-gpu/len16k-attn |
| 16k | global-attn24-pool | CPU | 4.87x | 1.000 | ar004-t006-t009/len16k-attn24-pool |
| 16k | global-attn24-pool-gpu | GPU | 12.78x | 1.000 | ar004-t015-matrix/len16k-attn24-pool-gpu |
| 16k | head | CPU | 3.83x | 0.000 | ar004-t002-final/len16k-head |
| 16k | head | GPU | 11.56x | 0.000 | ar004-t012-gpu/len16k-head |
| 32k | attn | CPU | 12.41x | 0.250 | ar004-t002-final/len32k-attn |
| 32k | global-attn24-gpu | GPU | 60.96x | 0.250 | ar004-t015-matrix/len32k-attn24-gpu |
| 32k | global-attn24-pool | CPU | 16.93x | 0.250 | ar004-t015-matrix/len32k-attn24-pool |
| 32k | head | CPU | 13.10x | 0.000 | ar004-t002-final/len32k-head |
| 4k | attn | CPU | 2.46x | 0.444 | ar004-t002-final/len4k-attn |
| 4k | global-attn24-pool | CPU | 1.90x | 1.000 | ar004-t015-matrix/len4k-attn24-pool |
| 4k | head | CPU | 2.23x | 0.000 | ar004-t002-final/len4k-head |
| 8021 | mn-global-attn24 | CPU | 2.54x | 0.000 | ar004-t006-t009/mn-attn24 |
| 8021 | mn-qg | CPU | 2.82x | 0.000 | ar004-t006-t009/mn-qg |
| 8k | attn | CPU | 2.61x | 0.444 | ar004-t002-final/len8k-attn |
| 8k | attn | GPU | 4.92x | 0.571 | ar004-t012-gpu/len8k-attn |
| 8k | global-attn24-pool | CPU | 2.21x | 1.000 | ar004-t006-t009/len8k-attn24-pool |
| 8k | head | CPU | 1.27x | 0.000 | ar004-t002-final/len8k-head |
