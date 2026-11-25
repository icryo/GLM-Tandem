#!/bin/bash
# runx.sh <script.py> [ENV=VAL ...]
# Run a harness script (experiments/tpmtp_x.py) as rank 0 on R0 and rank 1 on R1 with the patched MLX and the
# server's fence mode, then print the matching lines of rank 0's log. Copies nothing. The regular service must
# be stopped, and only one run may be in flight: each run loads the model on both Macs.
#   R0, R1, COORD_ADDR, IBV_DEVICES, MODEL_PATH, DIR as for servex.sh;  GREP = regexp of log lines to print
: ${R0:?} ${R1:?} ${COORD_ADDR:?} ${IBV_DEVICES:?} ${MODEL_PATH:?} ${DIR:?}
S=$1; shift; ENVS="$*"; PORT=$((48000 + RANDOM % 1000)); P=$DIR/scripts/pyx
C="MODEL_PATH=$MODEL_PATH TP_HCROWS=1 TP_MOEROWS=1 MLX_FENCE_MODE=1 MLX_METAL_FAST_SYNCH=1 MLX_MAX_MB_PER_BUFFER=20000 MLX_MAX_OPS_PER_BUFFER=100000 MLX_WORLD_SIZE=2 MLX_JACCL_COORDINATOR=$COORD_ADDR:$PORT MLX_IBV_DEVICES=$IBV_DEVICES $ENVS"
ssh $R0 "cd $DIR/server && $C MLX_RANK=0 nohup $P $S > /tmp/tp_r0.log 2>&1 < /dev/null &"
t=$(date +%s); while [ $(( $(date +%s) - t )) -lt 3 ]; do :; done
ssh $R1 "cd $DIR/server && $C MLX_RANK=1 timeout ${TMO:-600} $P $S > /tmp/tp_r1.log 2>&1; tail -2 /tmp/tp_r1.log | cut -c1-300"
t=$(date +%s); while ssh $R0 "pgrep -f 'python3 $S' >/dev/null" && [ $(( $(date +%s) - t )) -lt 20 ]; do :; done
ssh $R0 "pkill -f 'python3 $S'; grep -a -E '${GREP:-pipecost|prefillcost|wf|kpass|rcheck|Traceback|Error}' /tmp/tp_r0.log | cut -c1-300 | tail -${TAILN:-12}"
