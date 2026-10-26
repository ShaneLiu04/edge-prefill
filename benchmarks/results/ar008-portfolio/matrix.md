# AR008 combination matrix

model `6a1a2eb6d15622bf...`, quality floor 0.95, phrase_weight 0.5.

| dataset | combo | cases | q mean | q min | TTFT ms | vs dense | selected |
|---|---|---|---|---|---|---|---|
| deep/len32k | dense | 4 | 1.000 | 1.000 | 852060 | 1.0x | 31077 |
| deep/len32k | gate-p16 | 4 | 0.250 | 0.000 | 63801 | 13.4x | 4109 |
| deep/len32k | gate-p24-local32 | 4 | 1.000 | 1.000 | 66772 | 12.8x | 4109 |
| deep/len32k | selection | 4 | 0.875 | 0.500 | 23145 | 36.8x | 1565 |
| deep/multineedle | dense | 6 | 1.000 | 1.000 | 91969 | 1.0x | 7857 |
| deep/multineedle | gate-p16 | 6 | 1.000 | 1.000 | 38165 | 2.4x | 1985 |
| deep/multineedle | gate-p24-local32 | 6 | 1.000 | 1.000 | 16694 | 5.5x | 1985 |
| deep/multineedle | selection | 6 | 1.000 | 1.000 | 8211 | 11.2x | 401 |
| explen/len16k | dense | 4 | 1.000 | 1.000 | 302675 | 1.0x | 15576 |
| explen/len16k | gate-p16 | 4 | 1.000 | 1.000 | 57254 | 5.3x | 3912 |
| explen/len16k | gate-p24-local32 | 4 | 1.000 | 1.000 | 60088 | 5.0x | 3912 |
| explen/len16k | selection | 4 | 1.000 | 1.000 | 15857 | 19.1x | 792 |
| explen/len4k | dense | 4 | 1.000 | 1.000 | 66306 | 1.0x | 3946 |
| explen/len4k | gate-p16 | 4 | 1.000 | 1.000 | 18504 | 3.6x | 1002 |
| explen/len4k | gate-p24-local32 | 4 | 1.000 | 1.000 | 14537 | 4.6x | 1002 |
| explen/len4k | selection | 4 | 1.000 | 1.000 | 5514 | 12.0x | 322 |
| explen/len8k | dense | 4 | 1.000 | 1.000 | 141922 | 1.0x | 7812 |
| explen/len8k | gate-p16 | 4 | 1.000 | 1.000 | 25697 | 5.5x | 1972 |
| explen/len8k | gate-p24-local32 | 4 | 1.000 | 1.000 | 25137 | 5.6x | 1972 |
| explen/len8k | selection | 4 | 1.000 | 1.000 | 5689 | 24.9x | 388 |

## Routed combinations

- deep/len32k -> **gate-p24-local32** (floor-met, q=1.0, 66772 ms)
- deep/multineedle -> **selection** (floor-met, q=1.0, 8211 ms)
- explen/len16k -> **selection** (floor-met, q=1.0, 15857 ms)
- explen/len4k -> **selection** (floor-met, q=1.0, 5514 ms)
- explen/len8k -> **selection** (floor-met, q=1.0, 5689 ms)
