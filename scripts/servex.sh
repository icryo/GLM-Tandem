#!/bin/bash
# servex.sh start|stop [ENV=VAL ...]
# Start the server by hand on both Macs (API on rank 0, port 8010) with the launcher's settings, without copying
# anything. For measurements with the regular service stopped. Extra ENV=VAL pairs are appended and win.
#   R0, R1        ssh targets of the two Macs (R0 runs rank 0)
#   COORD_ADDR    R0's address on the Thunderbolt link;  IBV_DEVICES  RDMA device map;  MODEL_PATH  checkpoint
#   DIR           directory with server/*.py and scripts/pyx on both Macs
#   SCRIPT        server file to run (default tpserve.py), e.g. an experimental copy
: ${R0:?} ${R1:?} ${COORD_ADDR:?} ${IBV_DEVICES:?} ${MODEL_PATH:?} ${DIR:?}
cmd=$1; shift; ENVS="$*"; S=${SCRIPT:-tpserve.py}; P=$DIR/scripts/pyx; PORT=$((48000 + RANDOM % 1000))
wait_s() { t=$(date +%s); while [ $(( $(date +%s) - t )) -lt $1 ]; do :; done; }
for h in $R0 $R1; do ssh $h 'pkill -f "python3 tpserve"; true' 2>/dev/null; done
[ "$cmd" = stop ] && exit 0
wait_s 3
C="MODEL_PATH=$MODEL_PATH TP_CONC=8 TP_PROMPT_CONC=1 TP_MEMLOG=60 MLX_FENCE_MODE=1 MLX_METAL_FAST_SYNCH=1 MLX_MAX_MB_PER_BUFFER=20000 MLX_MAX_OPS_PER_BUFFER=100000 MLX_WORLD_SIZE=2 MLX_JACCL_COORDINATOR=$COORD_ADDR:$PORT MLX_IBV_DEVICES=$IBV_DEVICES $ENVS"
ssh $R0 "cd $DIR/server && $C MLX_RANK=0 nohup $P $S > /tmp/tpserve_r0.log 2>&1 < /dev/null &"
wait_s 3
ssh $R1 "cd $DIR/server && $C MLX_RANK=1 nohup $P $S > /tmp/tpserve_r1.log 2>&1 < /dev/null & sleep 2"
for i in $(seq 1 120); do ssh $R0 'curl -s -m2 localhost:8010/v1/models >/dev/null' && { echo "up ($S, rank 0 on $R0)"; exit 0; }; wait_s 3; done
echo "NOT UP after 360 s"; ssh $R0 'tail -5 /tmp/tpserve_r0.log'; ssh $R1 'tail -5 /tmp/tpserve_r1.log'; exit 1
