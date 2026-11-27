#!/bin/sh
cd /home/a/mlx-warm-fence/omlx5/scripts
run() { ./gatewait.sh 900 || { echo "SKIPPED $1 (gateway busy)"; return; }
        echo "== $1"; ssh mac2 "cd /tmp && timeout 90 $2 2>&1 | grep -v -i -E 'warning|falling back|Symbol not|Referenced from|Expected in'"; }
run "quant_matmul, MLX 0.32.4 (pyx)"     "~/llm-stack/cluster/tp/pyx -P /tmp/kbench_qmmcheck.py /tmp/kbench_qmm_0324.npz"
run "quant_matmul, MLX 0.32.5 (pyx5)"    "~/llm-stack/cluster/tp/pyx5 -P /tmp/kbench_qmmcheck.py /tmp/kbench_qmm_0325.npz"
echo "== compare saved outputs 0.32.4 vs 0.32.5"
ssh mac2 "~/llm-stack/cluster/tp/pyx -P /tmp/kbench_npzcmp.py /tmp/kbench_qmm_0324.npz /tmp/kbench_qmm_0325.npz"
echo "ALL DONE"
