#!/bin/bash
# n26check.sh suite | soak MINUTES     Correctness suite and mixed soak against the two-Mac server on :8010
# (straight to the model server, not through the gateway, so nothing lands in the request log or captures).
H=a@192.168.1.252
LOGF=${LOGF:-'~/llm-stack/logs/tp-cluster.log'}     # standalone server: LOGF=/tmp/tpserve_r0.log
U=http://localhost:8010/v1/chat/completions
case "$1" in
suite)
ssh $H "cd ~/llm-stack/cluster/tp && P=/Users/a/.omlx/bin/omlx-cluster-python; L=$LOGF; n0=\$(wc -l < \$L | tr -d ' ')
\$P validate.py 2>&1 | tail -1 | sed 's/^/validate: /'; \$P stress.py 2>&1 | tail -1 | sed 's/^/stress: /'; \$P vistest.py 2>&1 | tail -1 | sed 's/^/vis: /'
for i in 1 2; do \$P conctest.py 2>&1 | tail -1 | sed 's/^/conc: /'; done
( \$P conctest.py 2>&1 | tail -1 | sed 's/^/conc (trio): /' ) & ( \$P stress.py 2>&1 | tail -1 | sed 's/^/stress (trio): /' ) & ( \$P vistest.py 2>&1 | tail -1 | sed 's/^/vis (trio): /' ) & wait
\$P validate.py 2>&1 | tail -1 | sed 's/^/validate: /'
\$P stoptest.py 2>&1 | tail -1 | sed 's/^/stop: /'; SEEDED=1 \$P stoptest.py 2>&1 | tail -1 | sed 's/^/stop seeded: /'
EXACT=0 \$P seedtest.py 2>&1 | tail -1 | sed 's/^/seed (EXACT=0): /'
( ROUNDS=2 CTX=13000 \$P pretest.py 2>&1 | tail -1 | sed 's/^/preempt: /' ); ( CANCEL=1 ROUNDS=2 CTX=13000 \$P pretest.py 2>&1 | tail -1 | sed 's/^/preempt cancel: /' )
CTX=13500 FLOOD=20 \$P cachetest.py 2>&1 | tail -2 | sed 's/^/cache: /'
URL=$U MODEL=default_model \$P rewindconv.py 2>&1 | tail -2 | sed 's/^/rewind: /'
URL=$U MODEL=default_model TARGET=110000 TURNS=9 JOB=1 \$P editconv.py 2>&1 | tail -3 | sed 's/^/editconv 110k: /'
\$P validate_gw.py 2>&1 | tail -1 | sed 's/^/validate through the gateway: /'
echo \"tracebacks since start of suite: \$(tail -n +\$((n0 + 1)) \$L | grep -c Traceback) (BrokenPipe: \$(tail -n +\$((n0 + 1)) \$L | grep -c BrokenPipeError))\"
tail -n +\$((n0 + 1)) \$L | grep -E 'fatal|loop pause|Insufficient Memory|index scores not|fusion not' | sort | uniq -c | head -5
tail -n +\$((n0 + 1)) \$L | grep 'tp-mem r0' | awk '{if (\$5 > m) m = \$5} END {print \"max active GiB rank 0:\", m}'
echo \"server processes: \$(pgrep -f 'python3 tpserve' | wc -l | tr -d ' ')\"; curl -s -m 5 localhost:8010/health; echo"
ssh mac2 'echo "rank 1: processes $(pgrep -f "python3 tpserve" | wc -l | tr -d " "), tracebacks $(grep -c Traceback /tmp/tpserve_r1.log), max active $(grep "tp-mem r1" /tmp/tpserve_r1.log | awk "{if (\$5 > m) m = \$5} END {print m}") GiB"; grep -E "fatal|Insufficient Memory|index scores|gate/up" /tmp/tpserve_r1.log | sort | uniq -c | head -5'
;;
soak)
END=$(( $(date +%s) + 60 * ${2:-12} ))
ssh $H "cd ~/llm-stack/cluster/tp && P=/Users/a/.omlx/bin/omlx-cluster-python; L=$LOGF; n0=\$(wc -l < \$L | tr -d ' '); r=0
while [ \$(date +%s) -lt $END ]; do r=\$((r+1))
  ( \$P conctest.py 2>&1 | tail -1 | sed 's/^/conc: /' ) & ( \$P stress.py 2>&1 | tail -1 | sed 's/^/stress: /' ) & ( \$P vistest.py 2>&1 | tail -1 | sed 's/^/vis: /' ) &
  ( CANCEL=\$((r % 2)) ROUNDS=2 CTX=\$((4000 + 9000 * (r % 3))) \$P pretest.py 2>&1 | tail -1 | sed 's/^/preempt: /' ) &
  ( KIND=image N=4 W=2 \$P smallbench.py 2>&1 | tail -1 | cut -c1-60 | sed 's/^/small: /' ) &
  ( CONV_TOK=\$((60000 + 80000 * (r % 2))) TURNS=3 \$P eqvis.py 2>&1 | grep -c 'recall ok' | sed 's/^/eqvis recall ok (of 7): /' ) & wait
  \$P validate.py 2>&1 | tail -1 | sed 's/^/validate: /'
  [ \$((r % 3)) = 0 ] && CTX=13500 FLOOD=20 \$P cachetest.py 2>&1 | tail -2
done; echo rounds \$r; date
echo \"tracebacks: \$(tail -n +\$((n0 + 1)) \$L | grep -c Traceback) (BrokenPipe: \$(tail -n +\$((n0 + 1)) \$L | grep -c BrokenPipeError))\"
tail -n +\$((n0 + 1)) \$L | grep -E 'Error' | grep -v BrokenPipe | sort | uniq -c | head -5
tail -n +\$((n0 + 1)) \$L | grep -E 'fatal|loop pause|Insufficient Memory' | sort | uniq -c | head -5
tail -n +\$((n0 + 1)) \$L | grep 'tp-mem r0' | awk '{if (\$5 > m) m = \$5; p = \$NF} END {print \"max active GiB rank 0:\", m, \" last pressure level:\", p}'
tail -n +\$((n0 + 1)) \$L | grep 'Prompt Cache' | tail -1
echo \"server processes: \$(pgrep -f 'python3 tpserve' | wc -l | tr -d ' ')\"; curl -s -m 5 localhost:8010/health; echo"
ssh mac2 'echo "rank 1: processes $(pgrep -f "python3 tpserve" | wc -l | tr -d " "), tracebacks $(grep -c Traceback /tmp/tpserve_r1.log), max active $(grep "tp-mem r1" /tmp/tpserve_r1.log | awk "{if (\$5 > m) m = \$5} END {print m}") GiB"'
;;
hunt)
# hunt ROUNDS: many generations of every kind, to catch a hang (a round is ~110 generations). A request that
# does not come back within its timeout fails its script; the server's own watchdog would exit after 150 s.
ssh $H "cd ~/llm-stack/cluster/tp && P=/Users/a/.omlx/bin/omlx-cluster-python; L=$LOGF; n0=\$(wc -l < \$L | tr -d ' '); t0=\$(date +%s)
for r in \$(seq 1 ${2:-6}); do
  N=24 MT=160 \$P /tmp/fixed.py 2>&1 | awk '{n++; s+=\$3} END {printf \"round '\$r' single: %d generations, mean %.1f tok/s\n\", n, s/n}'
  ( \$P conctest.py 2>&1 | tail -1 | sed 's/^/  conc: /' ) & ( \$P stress.py 2>&1 | tail -1 | sed 's/^/  stress: /' ) & ( KIND=image N=8 W=2 \$P smallbench.py 2>&1 | tail -1 | cut -c1-70 | sed 's/^/  small image: /' ) & ( KIND=text N=8 W=2 \$P smallbench.py 2>&1 | tail -1 | cut -c1-70 | sed 's/^/  small text: /' ) & wait
  ROUNDS=6 \$P stoptest.py 2>&1 | tail -1 | sed 's/^/  stop: /'; ( CANCEL=1 ROUNDS=2 CTX=9000 \$P pretest.py 2>&1 | tail -1 | sed 's/^/  preempt+cancel: /' )
  \$P validate.py 2>&1 | tail -1 | sed 's/^/  validate: /'
done
echo \"hunt took \$(( \$(date +%s) - t0 )) s; server processes: \$(pgrep -f 'python3 tpserve' | wc -l | tr -d ' '); tracebacks: \$(tail -n +\$((n0 + 1)) \$L | grep -c Traceback) (BrokenPipe: \$(tail -n +\$((n0 + 1)) \$L | grep -c BrokenPipeError)); fatal/pause lines: \$(tail -n +\$((n0 + 1)) \$L | grep -cE 'fatal|loop pause|Insufficient Memory')\"; curl -s -m 5 localhost:8010/health; echo"
ssh mac2 'echo "rank 1: processes $(pgrep -f "python3 tpserve" | wc -l | tr -d " "), tracebacks $(grep -c Traceback /tmp/tpserve_r1.log)"'
;;
*) echo "usage: n26check.sh suite | soak MINUTES | hunt ROUNDS";;
esac
