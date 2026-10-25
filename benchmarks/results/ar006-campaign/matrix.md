# AR006 campaign matrix

Net speedup = dense / mode TTFT, both measured in this session.
quality is the mean over cases; q < 0.9 on a bound row fails the gate.

| dataset | mode | tokens | sel | ttft ms | dense ms | speedup | q | src |
|---|---|---|---|---|---|---|---|---|
| len16k | bound-p16 | 15576 | 3912 | 23700.2 | 161754.9 | 6.83x | 1.0 | attention-bound |
| len16k | bound-p24 | 15576 | 3912 | 23624.1 | 161754.9 | 6.85x | 1.0 | attention-bound |
| len16k | qg-2d(AR005) | 15576 | 3912 | 25120.4 | 161754.9 | 6.44x | 1.0 | embedding-2d |
| len16k | scalar | 15576 | 3912 | 22542.0 | 161754.9 | 7.18x | 1.0 |  |
| len32k | bound-p16 | 31080 | 4110 | 40943.0 | 383334.2 | 9.36x | 0.0 | attention-bound |
| len32k | bound-p24 | 31080 | 4110 | 62632.6 | 383334.2 | 6.12x | 0.25 | attention-bound |
| len32k | bound-p24+local32 | 31080 | 4110 | 67909.5 | 383334.2 | 5.64x | 1.0 | attention-bound |
| len32k | scalar | 31080 | 4110 | 22795.7 | 383334.2 | 16.82x | 1.0 |  |
| len4k | bound-p16 | 3947 | 1003 | 11967.4 | 50290.0 | 4.2x | 1.0 | attention-bound |
| len4k | qg-2d(AR005) | 3947 | 1003 | 28986.7 | 50290.0 | 1.73x | 1.0 | embedding-2d |
| len4k | scalar | 3947 | 1003 | 6223.9 | 50290.0 | 8.08x | 1.0 |  |
| len8k | bound-p16 | 7809 | 1975 | 11648.0 | 109306.3 | 9.38x | 1.0 | attention-bound |
| len8k | qg-2d(AR005) | 7809 | 1975 | 12463.2 | 109306.3 | 8.77x | 1.0 | embedding-2d |
| len8k | scalar | 7809 | 1975 | 11232.1 | 109306.3 | 9.73x | 1.0 |  |
| multineedle | bound-p16 | 7857 | 1985 | 17052.3 | 64250.6 | 3.77x | 0.833 | attention-bound |
| multineedle | scalar | 7857 | 1985 | 22717.8 | 64250.6 | 2.83x | 0.667 |  |
