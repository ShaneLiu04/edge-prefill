# AR007 budget sweep

model sha256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`, quality floor 0.95, margin 1.0, repeats 1.

| dataset | length | budget | signal | selected | q | min q | ttft ms |
|---|---|---|---|---|---|---|---|
| explen/len4k | 3962 | 0.05 | lexical | 210 | 0.500 | 0.000 | 3022 |
| explen/len4k | 3962 | 0.08 | lexical | 322 | 1.000 | 1.000 | 3696 |
| explen/len4k | 3962 | 0.12 | lexical | 482 | 1.000 | 1.000 | 3635 |
| explen/len8k | 7826 | 0.05 | lexical | 388 | 1.000 | 1.000 | 3349 |
| explen/len8k | 7826 | 0.08 | lexical | 644 | 1.000 | 1.000 | 5881 |
| explen/len8k | 7826 | 0.12 | lexical | 956 | 1.000 | 1.000 | 7667 |
| explen/len16k | 15603 | 0.05 | lexical | 792 | 1.000 | 1.000 | 6993 |
| explen/len16k | 15603 | 0.08 | lexical | 1256 | 1.000 | 1.000 | 11002 |
| explen/len16k | 15603 | 0.12 | lexical | 1880 | 0.500 | 0.000 | 15748 |
| deep/len32k | 31125 | 0.05 | lexical | 1565 | 0.893 | 0.571 | 13683 |
| deep/len32k | 31125 | 0.08 | lexical | 2509 | 0.667 | 0.000 | 14812 |
| deep/len32k | 31125 | 0.12 | lexical | 3749 | 0.750 | 0.000 | 26940 |

## Recommendation

Quality is not monotone in the budget on this task family, so the rule does not walk up from the smallest budget: among the cells that meet the floor it keeps those within `--speed-tolerance` of the fastest and takes the cheapest, which breaks timer-noise ties by budget. `floor-missed` marks lengths where no cell reached the floor; those keep their best measured quality and no margin, because a larger budget can be worse.

- 3962 tokens -> budget 0.08 (signal lexical, q=1.0, floor-met)
- 7826 tokens -> budget 0.05 (signal lexical, q=1.0, floor-met)
- 15603 tokens -> budget 0.05 (signal lexical, q=1.0, floor-met)
- 31125 tokens -> budget 0.05 (signal lexical, q=0.893, floor-missed)
