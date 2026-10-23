#!/bin/bash
# Deep-dive datasets: 32K long context + multi-needle robustness.
set -eu
NI=/mnt/d/prefill/benchmarks/data/niah
D=/tmp/deep
mkdir -p $D/len32k/contexts $D/multineedle/contexts

# ---- 32K: four 8K contexts concatenated, needle from vault 4 ----
cat $NI/contexts/niah-01.txt $NI/contexts/niah-02.txt \
    $NI/contexts/niah-03.txt $NI/contexts/niah-04.txt \
    > $D/len32k/contexts/niah-32k.txt
cat > $D/len32k/index.tsv <<'EOF'
id	task	context_path	question	answer
niah-32k	niah	contexts/niah-32k.txt	What is the secret passcode for vault 4?	VIOLET-PANTHER-7411
EOF

# ---- multi-needle 8K: vault 1 native + vault 2 at ~1/3 + vault 3 at ~2/3 ----
C=$NI/contexts/niah-01.txt
SZ=$(wc -c < $C)
T1=$((SZ / 3))
T2=$((2 * SZ / 3))
head -c $T1 $C > /tmp/mn_part1
head -c $((T2 - T1)) $C | tail -c $((T2 - T1)) > /tmp/mn_part2
tail -c $((SZ - T2)) $C > /tmp/mn_part3
cat /tmp/mn_part1 > $D/multineedle/contexts/mn-01.txt
printf ' The secret passcode for vault 2 is COBALT-HERON-7137.' >> $D/multineedle/contexts/mn-01.txt
cat /tmp/mn_part2 >> $D/multineedle/contexts/mn-01.txt
printf ' The secret passcode for vault 3 is AMBER-RAVEN-7274.' >> $D/multineedle/contexts/mn-01.txt
cat /tmp/mn_part3 >> $D/multineedle/contexts/mn-01.txt
cat > $D/multineedle/index.tsv <<'EOF'
id	task	context_path	question	answer
mn-v1	niah	contexts/mn-01.txt	What is the secret passcode for vault 1?	MAGENTA-FALCON-7000
mn-v2	niah	contexts/mn-01.txt	What is the secret passcode for vault 2?	COBALT-HERON-7137
mn-v3	niah	contexts/mn-01.txt	What is the secret passcode for vault 3?	AMBER-RAVEN-7274
EOF
wc -c $D/len32k/contexts/niah-32k.txt $D/multineedle/contexts/mn-01.txt
