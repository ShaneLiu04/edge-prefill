# AR010 combination matrix

quality floor = 0.95

## deep/len32k  (routed: selection)

| arm | q_mean | q_min | selected | TTFT ms |
|---|---:|---:|---:|---:|
| selection | 1.000 | 1.000 | 525 | 2259 |
| gate-p16 | 1.000 | 1.000 | 1165 | 5417 |
| gate-p24-local32 | 1.000 | 1.000 | 1165 | 5443 |
| selection-prior | 0.611 | 0.000 | 1565 | 7276 |

## deep/multifact8k  (routed: selection)

| arm | q_mean | q_min | selected | TTFT ms |
|---|---:|---:|---:|---:|
| selection | 0.836 | 0.429 | 431 | 2276 |
| gate-p16 | 0.836 | 0.429 | 431 | 2299 |
| gate-p24-local32 | 0.836 | 0.429 | 431 | 2312 |
| selection-prior | 0.836 | 0.429 | 443 | 2319 |

## deep/multifact8k-hard  (routed: gate-p16)

| arm | q_mean | q_min | selected | TTFT ms |
|---|---:|---:|---:|---:|
| gate-p16 | 0.667 | 0.087 | 445 | 2276 |
| gate-p24-local32 | 0.667 | 0.087 | 445 | 2280 |
| selection | 0.667 | 0.087 | 445 | 2305 |
| selection-prior | 0.667 | 0.087 | 445 | 2332 |

## deep/multineedle  (routed: selection)

| arm | q_mean | q_min | selected | TTFT ms |
|---|---:|---:|---:|---:|
| selection | 1.000 | 1.000 | 321 | 2675 |
| selection-prior | 1.000 | 1.000 | 401 | 2739 |
| gate-p24-local32 | 1.000 | 1.000 | 673 | 7043 |
| gate-p16 | 1.000 | 1.000 | 673 | 7611 |

## explen/len16k  (routed: selection)

| arm | q_mean | q_min | selected | TTFT ms |
|---|---:|---:|---:|---:|
| selection | 1.000 | 1.000 | 280 | 1251 |
| selection-prior | 1.000 | 1.000 | 792 | 3675 |
| gate-p16 | 1.000 | 1.000 | 1128 | 5089 |
| gate-p24-local32 | 1.000 | 1.000 | 1128 | 5351 |
| dense | 1.000 | 1.000 | 15576 | 132420 |

## explen/len4k  (routed: selection)

| arm | q_mean | q_min | selected | TTFT ms |
|---|---:|---:|---:|---:|
| selection | 1.000 | 1.000 | 226 | 1050 |
| selection-prior | 1.000 | 1.000 | 322 | 1491 |
| gate-p24-local32 | 1.000 | 1.000 | 354 | 1597 |
| gate-p16 | 1.000 | 1.000 | 354 | 1605 |
| dense | 1.000 | 1.000 | 3946 | 21394 |

## explen/len8k  (routed: selection)

| arm | q_mean | q_min | selected | TTFT ms |
|---|---:|---:|---:|---:|
| selection | 1.000 | 1.000 | 228 | 1039 |
| selection-prior | 1.000 | 1.000 | 388 | 1698 |
| gate-p24-local32 | 1.000 | 1.000 | 628 | 2789 |
| gate-p16 | 1.000 | 1.000 | 628 | 2991 |
| dense | 1.000 | 1.000 | 7812 | 50380 |

