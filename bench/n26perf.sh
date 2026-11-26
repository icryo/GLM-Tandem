#!/bin/bash
# n26perf.sh LABEL [parts]   Speed set against the standalone server on :8010 (window use).
# parts (default all): single conc edit small long turn
H=a@192.168.1.252; U=http://localhost:8010/v1/chat/completions; L=$1; PARTS=${2:-"single conc edit small long turn"}
has() { case " $PARTS " in *" $1 "*) return 0;; esac; return 1; }
echo "== $L $(date +%H:%M:%S)"
R="cd ~/llm-stack/cluster/tp && P=/Users/a/.omlx/bin/omlx-cluster-python; export URL=$U MODEL=default_model;"
has single && ssh $H "$R N=4 MT=900 \$P /tmp/fixed.py | awk '{n++; s+=\$3; printf \"%s \", \$3} END {printf \"| single stream T=0: mean %.1f tok/s\n\", s/n}'; TEMP=0.6 N=3 \$P bench_prose.py 2>&1 | tail -1 | cut -c1-120 | sed 's/^/prose T=0.6: /'"
has conc && ssh $H "$R \$P concbench.py 2>&1 | grep concurrent | cut -c1-150"
has edit && ssh $H "$R \$P bench_edit.py 2>&1 | tail -4 | cut -c1-110"
has small && ssh $H "$R KIND=image N=8 W=1 \$P smallbench.py 2>&1 | tail -1 | cut -c1-150; KIND=image N=8 W=2 \$P smallbench.py 2>&1 | tail -1 | cut -c1-150; KIND=text N=8 W=1 \$P smallbench.py 2>&1 | tail -1 | cut -c1-150"
has long && ssh $H "$R CTX=30000 N=1 GEN=500 ROUNDS=2 \$P longpair.py 2>&1 | tail -3 | cut -c1-170 | sed 's/^/70k + short: /'; CTX=52000 N=1 GEN=500 ROUNDS=2 \$P longpair.py 2>&1 | tail -3 | cut -c1-170 | sed 's/^/120k + short: /'"
has turn && ssh $H "$R CTX=52000 NEW=50 OUT=40 TURNS=8 \$P turnbench.py 2>&1 | grep -E 'cold|MEDIAN' | cut -c1-150 | sed 's/^/turns of ~115 new tokens at 120k: /'; CTX=52000 NEW=12 OUT=40 TURNS=8 \$P turnbench.py 2>&1 | grep -E 'MEDIAN' | cut -c1-150 | sed 's/^/turns of ~30 new tokens at 120k: /'"
ssh $H "grep 'tp-mem r0' /tmp/tpserve_r0.log | tail -1 | cut -c1-160"
echo "== end $L $(date +%H:%M:%S)"
