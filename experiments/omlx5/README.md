# oMLX native kernels for MLX 0.32.5 (Nov 26 evening, not built)

Why 0.32.5 lost long-context decode in window 7 (122 -> 94 tok/s at 143k): of oMLX 0.8.0rc1's four
native extensions only `sparse_index/_ext` fails to load under 0.32.5. The missing symbol is the
`metal::Device::get_library(name, CompileOptions, builder)` overload (0.32.5 moved `CompileOptions`
into `mlx::core::fast`), so it needs a recompile against the 0.32.5 headers, no source change.
Without it oMLX's GLM decode/verify skips `_fast_short_select` (`dsa_topk_indices`) and takes
matmul scores + argpartition (selected sets identical; ~40 us more per sparse layer at 2 rows,
~0.45 ms per pass at 141k: under 10% of the loss, the rest is the skipped fused path).

Prepared (mac2 `~/omlxsrc5/` = upstream tag v0.8.0rc1 at 454c90d35f, the installed version;
oMLX's own 0.32.5 port PR #5652 applies cleanly; the build must use MLX's bundled nanobind 3.1.0,
not the venv's 2.10.2, or oMLX's ABI check rejects the extension). The build command was refused
by the session's permission check and was not retried another way:

    cd ~/omlxsrc5/<tree> && PATH=~/mlxbuild/venv/bin:$PATH PYTHONPATH=~/mlxbuild5/mlx/python \
      OMLX_WITH_CUSTOM_KERNEL=1 CMAKE_ARGS="-Dnanobind_ROOT=~/mlxbuild5/mlx/build/temp.macosx-11.0-arm64-cpython-311/mlx.core/_deps/nanobind-src/cmake" \
      taskpolicy -b python setup.py build_ext --build-lib ~/omlx5ext/lib --build-temp ~/omlx5ext/tmp

Offline on mac2 (0.32.4 -> 0.32.5): bundled quant_matmul `affine_q4_qmm_t` bit-identical and same
speed; stock `mx.quantized_matmul` 12-15% faster at 128-2,048 rows (4096x4096 and 12544x4096),
but 2.2x slower and not bit-identical at 4096x4096 x 128 rows. Estimate with the extension
rebuilt: long-context decode back to 0.32.4 levels, net maybe +2-4% if large prompt projections
go to stock 0.32.5. Alternative: oMLX 0.8.4 (pins 0.32.5, rebuilt kernels) = whole app upgrade.

Scripts here: gate.sh / gatewait.sh (idle gate), indexfb.py (index selection fallback vs native),
qmmcheck.py / npzcmp.py / runqmm.sh / runall.sh (quantized matmul equality and speed), logs.
