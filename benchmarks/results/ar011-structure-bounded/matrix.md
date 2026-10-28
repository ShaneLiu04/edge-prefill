# AR011 combination matrix

global quality floor = 0.95; per-dataset floors are the AR010 measurement where it is below the global floor

## deep/len32k  (routed: ar011, floor 0.950/0.950)

| arm | q_mean | q_min | selected | struct | span | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| ar010 | 1.000 | 1.000 | 525 | 45 | 0 | 2371 |
| ar011 | 1.000 | 1.000 | 525 | 45 | 0 | 2329 |
| gate-p16 | 1.000 | 1.000 | 1165 | 45 | 0 | 5953 |

## deep/multifact8k  (routed: ar011, floor 0.836/0.429)

| arm | q_mean | q_min | selected | struct | span | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| ar010 | 0.836 | 0.429 | 431 | 47 | 0 | 1924 |
| ar011 | 0.836 | 0.429 | 431 | 47 | 0 | 2002 |
| gate-p16 | 0.836 | 0.429 | 431 | 47 | 0 | 1932 |

## deep/multifact8k-hard  (routed: ar011, floor 0.667/0.087)

| arm | q_mean | q_min | selected | struct | span | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| ar010 | 0.667 | 0.087 | 445 | 49 | 0 | 1997 |
| ar011 | 0.709 | 0.571 | 445 | 49 | 0 | 1952 |
| gate-p16 | 0.667 | 0.087 | 445 | 49 | 0 | 1933 |

## deep/multineedle  (routed: ar011, floor 0.950/0.950)

| arm | q_mean | q_min | selected | struct | span | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| ar011 | 1.000 | 1.000 | 247 | 49 | 36 | 1165 |
| ar010 | 1.000 | 1.000 | 321 | 49 | 0 | 1790 |
| gate-p16 | 1.000 | 1.000 | 673 | 49 | 0 | 3266 |
| dense | 1.000 | 1.000 | 7857 | 0 | 0 | 69133 |

## explen/len16k  (routed: ar011, floor 0.950/0.950)

| arm | q_mean | q_min | selected | struct | span | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| ar010 | 1.000 | 1.000 | 280 | 48 | 0 | 1444 |
| ar011 | 1.000 | 1.000 | 280 | 48 | 0 | 1387 |
| gate-p16 | 1.000 | 1.000 | 1128 | 48 | 0 | 5480 |
| dense | 1.000 | 1.000 | 15576 | 0 | 0 | 172620 |

## explen/len4k  (routed: ar011, floor 0.950/0.950)

| arm | q_mean | q_min | selected | struct | span | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| ar011 | 1.000 | 1.000 | 146 | 50 | 0 | 755 |
| ar010 | 1.000 | 1.000 | 226 | 50 | 0 | 1046 |
| gate-p16 | 1.000 | 1.000 | 354 | 50 | 0 | 1631 |
| dense | 1.000 | 1.000 | 3946 | 0 | 0 | 30500 |

## explen/len8k  (routed: ar011, floor 0.950/0.950)

| arm | q_mean | q_min | selected | struct | span | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| ar011 | 1.000 | 1.000 | 188 | 52 | 0 | 1028 |
| ar010 | 1.000 | 1.000 | 228 | 52 | 0 | 1091 |
| gate-p16 | 1.000 | 1.000 | 628 | 52 | 0 | 3036 |
| dense | 1.000 | 1.000 | 7812 | 0 | 0 | 69671 |

