#!/bin/zsh
# Foreground launcher for the two-Mac tensor-parallel GLM server. Run on the first Mac (rank 0).
# Starts rank 1 on the second Mac, runs rank 0 here, and takes rank 1 down when rank 0 stops.
# API: OpenAI-style on :8010 (TP_PORT).
#
# Required:
#   PEER          ssh target of the second Mac over the Thunderbolt link, e.g. user@10.0.0.2
#   COORD_ADDR    this Mac's address on the Thunderbolt link, e.g. 10.0.0.1
#   IBV_DEVICES   path to the RDMA device map (same file as for oMLX's cluster mode),
#                 e.g. a file containing [[null,"rdma_en3"],["rdma_en3",null]]
#   MODEL_PATH    checkpoint directory, present at the same path on both Macs
# Optional:
#   DIR           directory holding server/*.py and scripts/pyx on both Macs (default: this repo)
#   SSH_KEY       key for the ssh hops
set -e
: ${PEER:?} ${COORD_ADDR:?} ${IBV_DEVICES:?} ${MODEL_PATH:?}
DIR=${DIR:-${0:A:h:h}}
P=$DIR/scripts/pyx
KEY=(${SSH_KEY:+-i $SSH_KEY})
OPTS=($KEY -o BatchMode=yes -o ConnectTimeout=5)
# Rank 0 goes through a local ssh hop: macOS Local Network privacy blocks LAN traffic for
# processes started by launchd, but not for sshd sessions. Harmless when run from a terminal.
SELF=(ssh $OPTS $USER@127.0.0.1)
REMOTE=(ssh $OPTS $PEER)
PORT=$((48000 + RANDOM % 1000))
# MLX_FENCE_MODE=1 needs the patched MLX build; remove it to run on stock MLX.
# TP_CONC: requests decoded together. Prompts are processed one at a time (batched prompt processing crashes).
# TP_SEEDED=2: requests that send a seed share decode steps with others (1 = they decode alone and repeat exactly).
ENVS=(MODEL_PATH=$MODEL_PATH TP_CONC=8 TP_PROMPT_CONC=1 TP_SEEDED=2 MLX_FENCE_MODE=1 MLX_METAL_FAST_SYNCH=1
      MLX_MAX_MB_PER_BUFFER=20000 MLX_MAX_OPS_PER_BUFFER=100000 MLX_WORLD_SIZE=2
      MLX_JACCL_COORDINATOR=$COORD_ADDR:$PORT MLX_IBV_DEVICES=$IBV_DEVICES)
set +e
$SELF 'pkill -f "python3 tpserve.py"; true' 2>/dev/null
$REMOTE 'pkill -f "python3 tpserve.py"; true' 2>/dev/null
sleep 2
child=0; peer=0; mon=0
cleanup() {
  [ $mon -ne 0 ] && kill $mon 2>/dev/null      # the peer monitor below must not outlive this script's server
  [ $peer -ne 0 ] && kill $peer 2>/dev/null
  $SELF 'pkill -f "python3 tpserve.py"; true' 2>/dev/null
  [ $child -ne 0 ] && kill $child 2>/dev/null
  $REMOTE 'pkill -f "python3 tpserve.py"; true' 2>/dev/null
}
trap 'cleanup; exit 0' TERM INT
$SELF "cd $DIR/server && exec env ${ENVS[*]} MLX_RANK=0 $P tpserve.py" &
child=$!
sleep 3
# Keep this ssh session open for the life of rank 1: once it ends the process is no longer an
# sshd child and Local Network privacy cuts its link to rank 0.
$REMOTE "cd $DIR/server && exec env ${ENVS[*]} MLX_RANK=1 $P tpserve.py > /tmp/tpserve_r1.log 2>&1" < /dev/null &
peer=$!
# If rank 1 goes away, rank 0 would wait on it until its own watchdog fires (TP_WATCHDOG_S, 150 s). Stop rank 0
# as soon as rank 1's session ends so a supervisor restarts both. Does nothing once this script has exited.
me=$$
{ while kill -0 $peer 2>/dev/null; do sleep 2; done; kill -0 $me 2>/dev/null && $SELF 'pkill -f "python3 tpserve.py"; true' 2>/dev/null; } &
mon=$!
wait $child
code=$?
cleanup
exit $code
