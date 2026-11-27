# Tile-fill prompt expert kernel ("exact codes"): the sorted gather_qmm of oMLX's tensor-unit db schedule, with the
# weight tile filled by 4-bit codes OR-ed into bf16 0x4300 (= 128 + q, exact, two weights per integer op) instead of
# bf16(scale * q + bias) per weight. Each 64-deep K step is one quantization group: its MMAs go into a fresh partial,
# then out += scale * partial + (bias - 128 * scale) * rowsum_g(x), with rowsum_g(x) the fp32 sum of the row's 64
# activations of that group (a small pre-pass). The code products are exact, so the result is closer to the exact
# product than the stock path (which rounds every weight to bf16), but NOT bit-identical to it.
#
# Builds on oMLX's m5_gather_qmm_nax (tile pre-pass, tile_of, Geo, NAX headers of the installed mlx), so it must run
# in the oMLX python (no -I). Modes (env MODE):
#   selftest  small canary vs a float32 reference (+ the stock path), every plan
#   acc       accuracy at the real per-Mac shapes vs an exact float64 reference (CPU), stock path vs this kernel
#   speed     gate (= up) and down calls at T tokens, stock tile kernel vs this kernel, per layer and x42
# Memory: ~1.4 GB of synthetic weights (+ activations), a few seconds of GPU per T.
import os, sys, time, math
import numpy as np
import mlx.core as mx
from omlx.patches import m5_gather_qmm_nax as NX

TF_HEADER = r"""
namespace omlx_gqmm {

STEEL_CONST int kTfPitch = 64 + 8;  // bf16 row pitch of a 64-deep code tile (as oMLX's BKP)

// Loader thread lid owns weight row lid / kTPR of the 64 x 64 tile and kVPT consecutive codes of it; it also
// carries its row's scale and bias of the step's group. store(): codes + 128 as bf16 (exact), and for the
// thread at column 0 the row's scale and (bias - 128 scale) in fp32.
template <typename T, typename G>
struct TfCodeLoader {
  STEEL_CONST int kVPT = G::kVPT;
  STEEL_CONST int kWords = kVPT / 8;
  static_assert(G::kBK == 64 && kWords * 8 == kVPT && kWords >= 1, "64-deep steps of whole words");
  const device uint32_t* src;
  const device T* sp;
  const device T* bp;
  const short row;
  const short col;
  uint32_t raw[kWords];
  float s;
  float b;

  METAL_FUNC TfCodeLoader(const device uint8_t* w_tile, const device T* s_tile, const device T* b_tile,
                          const int K, const uint lid) thread
      : row(short(lid / G::kTPR)), col(short((lid % G::kTPR) * kVPT)) {
    src = (const device uint32_t*)(w_tile + size_t(row) * (K / 2) + col / 2);
    sp = s_tile + size_t(row) * (K / 64);
    bp = b_tile + size_t(row) * (K / 64);
  }

  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* ptr = src + kb * 8;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      raw[i] = ptr[i];
    }
    s = float(sp[kb]);
    b = float(bp[kb]);
  }

  METAL_FUNC void store(threadgroup T* Ws, threadgroup float* sb) const thread {
    threadgroup uint4* o = (threadgroup uint4*)(Ws + row * kTfPitch + col);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      const uint32_t w = raw[i];
      const uint32_t lo = w & 0x0f0f0f0fu;
      const uint32_t hi = (w >> 4) & 0x0f0f0f0fu;
      o[i] = uint4(0x43004300u | (lo & 0xffu) | ((hi & 0xffu) << 16),
                   0x43004300u | ((lo >> 8) & 0xffu) | (((hi >> 8) & 0xffu) << 16),
                   0x43004300u | ((lo >> 16) & 0xffu) | (((hi >> 16) & 0xffu) << 16),
                   0x43004300u | (lo >> 24) | ((hi >> 24) << 16));
    }
    if (col == 0) {
      sb[row] = s;
      sb[kBN + row] = b - 128.0f * s;
    }
  }
};


// One 16 x 32 x 16 tensor op like BaseNAXFrag::mma; ACC false overwrites C (multiply mode, nothing copied in).
template <bool ACC, typename CType, typename AType, typename BType>
METAL_FUNC void tf_mma(
    thread BaseNAXFrag::dtype_frag_t<CType>& Cn0,
    thread BaseNAXFrag::dtype_frag_t<CType>& Cn1,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    const thread BaseNAXFrag::dtype_frag_t<BType>& Bn0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& Bn1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      ACC ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
          : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>, metal::remove_addrspace_t<decltype(ct_b)>, CType>();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 8; i++) {
    ct_a[i] = A[i];
    ct_b[i] = Bn0[i];
    ct_b[8 + i] = Bn1[i];
  }
  if constexpr (ACC) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 8; i++) {
      ct_c[i] = Cn0[i];
      ct_c[8 + i] = Cn1[i];
    }
  }
  gemm_op.run(ct_a, ct_b, ct_c);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 8; i++) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[8 + i];
  }
}

// TFM 4: no codes, no epilogue: the weight tile is fma(q, scale, bias) in half precision (q exact via 0x6400,
// two weights an op), the activations are converted to half as they are loaded, half x half tensor ops.
template <typename T, typename G>
struct TfHalfLoader {
  STEEL_CONST int kVPT = G::kVPT;
  STEEL_CONST int kWords = kVPT / 8;
  const device uint32_t* src;
  const device T* sp;
  const device T* bp;
  const short row;
  const short col;
  uint32_t raw[kWords];
  half s;
  half b;
  METAL_FUNC TfHalfLoader(const device uint8_t* w_tile, const device T* s_tile, const device T* b_tile,
                          const int K, const uint lid) thread
      : row(short(lid / G::kTPR)), col(short((lid % G::kTPR) * kVPT)) {
    src = (const device uint32_t*)(w_tile + size_t(row) * (K / 2) + col / 2);
    sp = s_tile + size_t(row) * (K / 64);
    bp = b_tile + size_t(row) * (K / 64);
  }
  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* ptr = src + kb * 8;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      raw[i] = ptr[i];
    }
    s = half(float(sp[kb]));
    b = half(float(bp[kb]));
  }
  METAL_FUNC void store(threadgroup half* Ws) const thread {
    threadgroup uint4* o = (threadgroup uint4*)(Ws + row * kTfPitch + col);
    const half2 s2 = half2(s, s);
    const half2 b2 = half2(b, b);
    const half2 off = half2(1024.0h, 1024.0h);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      const uint32_t w = raw[i];
      const uint32_t lo = w & 0x0f0f0f0fu;
      const uint32_t hi = (w >> 4) & 0x0f0f0f0fu;
      const half2 v0 = as_type<half2>(0x64006400u | (lo & 0xffu) | ((hi & 0xffu) << 16)) - off;
      const half2 v1 = as_type<half2>(0x64006400u | ((lo >> 8) & 0xffu) | (((hi >> 8) & 0xffu) << 16)) - off;
      const half2 v2 = as_type<half2>(0x64006400u | ((lo >> 16) & 0xffu) | (((hi >> 16) & 0xffu) << 16)) - off;
      const half2 v3 = as_type<half2>(0x64006400u | (lo >> 24) | ((hi >> 24) << 16)) - off;
      o[i] = uint4(as_type<uint32_t>(fma(v0, s2, b2)), as_type<uint32_t>(fma(v1, s2, b2)),
                   as_type<uint32_t>(fma(v2, s2, b2)), as_type<uint32_t>(fma(v3, s2, b2)));
    }
  }
};

template <typename T, typename G, typename OT>
METAL_FUNC void tf_gather_db_half(
    const device T* x,
    const device uint8_t* w,
    const device T* scales,
    const device T* biases,
    const uint4 desc,
    const int y_col,
    device OT* y,
    const int N,
    const int K,
    threadgroup half* Ws,
    const uint sgid,
    const uint lane) {
  constexpr int kTile = kBN * kTfPitch;
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);
  const int K_g = K / 64;
  const int K_it = K / 64;
  const size_t w_row = size_t(expert) * N + y_col;
  TfHalfLoader<T, G> loader(w + w_row * (K / 2), scales + w_row * K_g, biases + w_row * K_g, K, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);
  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  const int base_row = row_start + max(0, min(m0, tile_rows - 1));
  const device T* xb = x + size_t(base_row) * K;
  const short2 sc = BaseNAXFrag::get_coord();
  int x_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      x_off[i][h] = r * K + sc.x;
    }
  }
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const int cn = int(sgid % kWN) * kSN;
  const threadgroup half* wsg = Ws + cn * kTfPitch;
  NAXTile<float, kTM, kTN> D;
  D.clear();
  if (loads) {
    loader.fetch(0);
    loader.store(Ws);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more && loads) {
      loader.fetch(kb + 1);
    }
    const threadgroup half* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < 64; kk1 += kSK) {
      NAXTile<half, kTN, 2> Btile;
      Btile.template load<half, kTfPitch, 1>(wb + kk1);
      const int k = kb * 64 + kk1;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        if (i < m_frags) {
          NAXTile<half, 1, 2> Atile;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const device T* xp = xb + x_off[i][h] + k;
            const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
            const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              Atile.frag_at(0, 0)[h * 4 + c] = half(float(a0[c]));
              Atile.frag_at(0, 1)[h * 4 + c] = half(float(a1[c]));
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < 2; kk++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j += 2) {
              BaseNAXFrag::mma(
                  D.frag_at(i, j),
                  D.frag_at(i, j + 1),
                  Atile.frag_at(0, kk),
                  metal::bool_constant<false>{},
                  Btile.frag_at(j, kk),
                  Btile.frag_at(j + 1, kk),
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
    if (more && loads) {
      loader.store(Ws + ((kb + 1) & 1) * kTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device OT* yb = y + size_t(row_start + m0) * N + y_col + cn;
  if (rows >= kSM) {
    D.store(yb, N);
  } else if (rows > 0) {
    D.store_safe(yb, N, short2(kSN, short(rows)));
  }
}

// oMLX's gather_db (double-buffered 64-deep steps, one barrier a step, activation fragments from device memory,
// 16-row fragments without rows skipped) with code tiles and a per-group epilogue.
template <typename T, typename G, typename OT, int TFM>
METAL_FUNC void tf_gather_db(
    const device T* x,
    const device float* xs,
    const device uint8_t* w,
    const device T* scales,
    const device T* biases,
    const uint4 desc,
    const int y_col,
    device OT* y,
    const int N,
    const int K,
    threadgroup T* Ws,
    threadgroup float* SB,
    const uint sgid,
    const uint lane) {
  constexpr int kTile = kBN * kTfPitch;
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);
  const int K_g = K / 64;
  const int K_it = K / 64;

  const size_t w_row = size_t(expert) * N + y_col;
  TfCodeLoader<T, G> loader(w + w_row * (K / 2), scales + w_row * K_g, biases + w_row * K_g, K, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);

  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  const int base_row = row_start + max(0, min(m0, tile_rows - 1));
  const device T* xb = x + size_t(base_row) * K;
  const device float* xsb = xs + size_t(base_row) * K_g;

  const short2 sc = BaseNAXFrag::get_coord();
  int x_off[kTM][2];
  int s_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      x_off[i][h] = r * K + sc.x;
      s_off[i][h] = r * K_g;
    }
  }
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const int cn = int(sgid % kWN) * kSN;
  const threadgroup T* wsg = Ws + cn * kTfPitch;

  NAXTile<float, kTM, kTN> D;
  D.clear();

  // One K step's MMAs into Pt (cleared first; TFM 1 accumulates into D directly).
  auto mmas = [&](thread NAXTile<float, kTM, kTN>& Pt, const int kb) {
    const threadgroup T* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < 64; kk1 += kSK) {
      NAXTile<T, kTN, 2> Btile;
      Btile.template load<T, kTfPitch, 1>(wb + kk1);
      const int k = kb * 64 + kk1;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        if (i < m_frags) {
          NAXTile<T, 1, 2> Atile;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const device T* xp = xb + x_off[i][h] + k;
            const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
            const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
              Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < 2; kk++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j += 2) {
              if (TFM == 3 && kk1 == 0 && kk == 0) {
                tf_mma<false, float, T, T>(Pt.frag_at(i, j), Pt.frag_at(i, j + 1), Atile.frag_at(0, kk),
                                           Btile.frag_at(j, kk), Btile.frag_at(j + 1, kk));
              } else {
                BaseNAXFrag::mma(
                    Pt.frag_at(i, j),
                    Pt.frag_at(i, j + 1),
                    Atile.frag_at(0, kk),
                    metal::bool_constant<false>{},
                    Btile.frag_at(j, kk),
                    Btile.frag_at(j + 1, kk),
                    metal::bool_constant<true>{});
              }
            }
          }
        }
      }
    }
  };
  // The group's sums are done: its scale, and its bias (minus 128 scale) times the rows' group sums.
  auto epi = [&](thread const NAXTile<float, kTM, kTN>& Pt, const int kb, const threadgroup float* sbb) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTM; i++) {
      if (i < m_frags) {
        const float xr0 = xsb[s_off[i][0] + kb];
        const float xr1 = xsb[s_off[i][1] + kb];
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kTN; j++) {
          const float4 s4 = *(const threadgroup float4*)(sbb + cn + 16 * j + sc.x);
          const float4 c4 = *(const threadgroup float4*)(sbb + kBN + cn + 16 * j + sc.x);
          STEEL_PRAGMA_UNROLL
          for (short c = 0; c < 4; c++) {
            D.frag_at(i, j)[c] = fma(s4[c], Pt.frag_at(i, j)[c], fma(c4[c], xr0, D.frag_at(i, j)[c]));
            D.frag_at(i, j)[4 + c] = fma(s4[c], Pt.frag_at(i, j)[4 + c], fma(c4[c], xr1, D.frag_at(i, j)[4 + c]));
          }
        }
      }
    }
  };
  constexpr int kSBN = TFM == 2 ? 3 : 2;   // scale buffers (the deferred epilogue reads the one before last)
  if (loads) {
    loader.fetch(0);
    loader.store(Ws, SB);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr (TFM == 2) {
    NAXTile<float, kTM, kTN> P0;
    NAXTile<float, kTM, kTN> P1;
    for (int kb = 0; kb < K_it; kb += 2) {
      // step kb into P0, then the epilogue of step kb - 1 (P1)
      if (loads) loader.fetch(kb + 1);
      P0.clear();
      mmas(P0, kb);
      if (kb > 0) epi(P1, kb - 1, SB + ((kb - 1) % kSBN) * (2 * kBN));
      if (loads) loader.store(Ws + ((kb + 1) & 1) * kTile, SB + ((kb + 1) % kSBN) * (2 * kBN));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // step kb + 1 into P1, then the epilogue of step kb (P0)
      const bool more = kb + 2 < K_it;
      if (more && loads) loader.fetch(kb + 2);
      P1.clear();
      mmas(P1, kb + 1);
      epi(P0, kb, SB + (kb % kSBN) * (2 * kBN));
      if (more && loads) loader.store(Ws + ((kb + 2) & 1) * kTile, SB + ((kb + 2) % kSBN) * (2 * kBN));
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    epi(P1, K_it - 1, SB + ((K_it - 1) % kSBN) * (2 * kBN));
  } else {
    for (int kb = 0; kb < K_it; kb++) {
      const bool more = kb + 1 < K_it;
      if (more && loads) {
        loader.fetch(kb + 1);
      }
      if constexpr (TFM == 1) {
        mmas(D, kb);
      } else {
        NAXTile<float, kTM, kTN> P;
        if constexpr (TFM != 3) P.clear();
        mmas(P, kb);
        epi(P, kb, SB + (kb & 1) * (2 * kBN));
      }
      if (more && loads) {
        loader.store(Ws + ((kb + 1) & 1) * kTile, SB + ((kb + 1) & 1) * (2 * kBN));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  device OT* yb = y + size_t(row_start + m0) * N + y_col + cn;
  if (rows >= kSM) {
    D.store(yb, N);
  } else if (rows > 0) {
    D.store_safe(yb, N, short2(kSN, short(rows)));
  }
}


// TFM 5 / 6: the code tile + group epilogue with every register tile sized to the simdgroup's live 16-row
// fragments (TL = 1 or 2, chosen per simdgroup; dead simdgroups only load). TFM 5 defers each group's epilogue
// by one step (two partials alternate, so a step's tensor ops never wait for the previous epilogue); TFM 6
// runs it right after the step's tensor ops.
template <typename T, typename G, typename OT, int TL, int TFM>
METAL_FUNC void tf_db_live(
    const device T* xb, const device float* xsb, thread TfCodeLoader<T, G>& loader, const bool loads,
    const thread int (&x_off)[kTM][2], const thread int (&s_off)[kTM][2], const bool live, const int K_it,
    const int cn, const short2 sc, threadgroup T* Ws, threadgroup float* SB, device OT* yb, const int N,
    const int rows) {
  constexpr int kTile = kBN * kTfPitch;
  constexpr int kSBN = 3;
  const threadgroup T* wsg = Ws + cn * kTfPitch;
  NAXTile<float, TL, kTN> D;
  D.clear();
  auto mmas = [&](thread NAXTile<float, TL, kTN>& Pt, const int kb) {
    const threadgroup T* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < 64; kk1 += kSK) {
      NAXTile<T, kTN, 2> Btile;
      Btile.template load<T, kTfPitch, 1>(wb + kk1);
      const int k = kb * 64 + kk1;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TL; i++) {
        NAXTile<T, 1, 2> Atile;
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; h++) {
          const device T* xp = xb + x_off[i][h] + k;
          const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
          const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
          STEEL_PRAGMA_UNROLL
          for (short c = 0; c < 4; c++) {
            Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
            Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < 2; kk++) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kTN; j += 2) {
            BaseNAXFrag::mma(Pt.frag_at(i, j), Pt.frag_at(i, j + 1), Atile.frag_at(0, kk),
                             metal::bool_constant<false>{}, Btile.frag_at(j, kk), Btile.frag_at(j + 1, kk),
                             metal::bool_constant<true>{});
          }
        }
      }
    }
  };
  auto epi = [&](thread const NAXTile<float, TL, kTN>& Pt, const int kb, const threadgroup float* sbb) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TL; i++) {
      const float xr0 = xsb[s_off[i][0] + kb];
      const float xr1 = xsb[s_off[i][1] + kb];
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTN; j++) {
        const float4 s4 = *(const threadgroup float4*)(sbb + cn + 16 * j + sc.x);
        const float4 c4 = *(const threadgroup float4*)(sbb + kBN + cn + 16 * j + sc.x);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c++) {
          D.frag_at(i, j)[c] = fma(s4[c], Pt.frag_at(i, j)[c], fma(c4[c], xr0, D.frag_at(i, j)[c]));
          D.frag_at(i, j)[4 + c] = fma(s4[c], Pt.frag_at(i, j)[4 + c], fma(c4[c], xr1, D.frag_at(i, j)[4 + c]));
        }
      }
    }
  };
  if (loads) {
    loader.fetch(0);
    loader.store(Ws, SB);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr (TFM == 5) {
    NAXTile<float, TL, kTN> P0;
    NAXTile<float, TL, kTN> P1;
    for (int kb = 0; kb < K_it; kb += 2) {
      if (loads) loader.fetch(kb + 1);
      if (live) {
        P0.clear();
        mmas(P0, kb);
        if (kb > 0) epi(P1, kb - 1, SB + ((kb - 1) % kSBN) * (2 * kBN));
      }
      if (loads) loader.store(Ws + ((kb + 1) & 1) * kTile, SB + ((kb + 1) % kSBN) * (2 * kBN));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const bool more = kb + 2 < K_it;
      if (more && loads) loader.fetch(kb + 2);
      if (live) {
        P1.clear();
        mmas(P1, kb + 1);
        epi(P0, kb, SB + (kb % kSBN) * (2 * kBN));
      }
      if (more && loads) loader.store(Ws + ((kb + 2) & 1) * kTile, SB + ((kb + 2) % kSBN) * (2 * kBN));
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (live) epi(P1, K_it - 1, SB + ((K_it - 1) % kSBN) * (2 * kBN));
  } else {
    for (int kb = 0; kb < K_it; kb++) {
      const bool more = kb + 1 < K_it;
      if (more && loads) loader.fetch(kb + 1);
      if (live) {
        NAXTile<float, TL, kTN> P;
        P.clear();
        mmas(P, kb);
        epi(P, kb, SB + (kb % kSBN) * (2 * kBN));
      }
      if (more && loads) loader.store(Ws + ((kb + 1) & 1) * kTile, SB + ((kb + 1) % kSBN) * (2 * kBN));
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (live) {
    if (rows >= TL * 16) {
      D.store(yb, N);
    } else {
      D.store_safe(yb, N, short2(kSN, short(rows)));
    }
  }
}

template <typename T, typename G, typename OT, int TFM>
METAL_FUNC void tf_gather_db_live(
    const device T* x, const device float* xs, const device uint8_t* w, const device T* scales,
    const device T* biases, const uint4 desc, const int y_col, device OT* y, const int N, const int K,
    threadgroup T* Ws, threadgroup float* SB, const uint sgid, const uint lane) {
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);
  const int K_g = K / 64;
  const int K_it = K / 64;
  const size_t w_row = size_t(expert) * N + y_col;
  TfCodeLoader<T, G> loader(w + w_row * (K / 2), scales + w_row * K_g, biases + w_row * K_g, K, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);
  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  const int base_row = row_start + max(0, min(m0, tile_rows - 1));
  const device T* xb = x + size_t(base_row) * K;
  const device float* xsb = xs + size_t(base_row) * K_g;
  const short2 sc = BaseNAXFrag::get_coord();
  int x_off[kTM][2];
  int s_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      x_off[i][h] = r * K + sc.x;
      s_off[i][h] = r * K_g;
    }
  }
  const int cn = int(sgid % kWN) * kSN;
  device OT* yb = y + size_t(row_start + m0) * N + y_col + cn;
  if (rows > 16) {
    tf_db_live<T, G, OT, 2, TFM>(xb, xsb, loader, loads, x_off, s_off, true, K_it, cn, sc, Ws, SB, yb, N, rows);
  } else {
    tf_db_live<T, G, OT, 1, TFM>(xb, xsb, loader, loads, x_off, s_off, rows > 0, K_it, cn, sc, Ws, SB, yb, N, rows);
  }
}

} // namespace omlx_gqmm
"""

TF_SOURCE = r"""
    using G = omlx_gqmm::Geo<BM, 64>;
    threadgroup T Ws[2 * omlx_gqmm::kBN * omlx_gqmm::kTfPitch];
    threadgroup float SB[6 * omlx_gqmm::kBN];
    uint4 desc;
    int y_col;
    if (!omlx_gqmm::tile_of<GX>(tiles, tile_count[0], threadgroup_position_in_grid, params[0], desc, y_col)) {
        return;
    }
    if constexpr (TFM == 5 || TFM == 6) {
        omlx_gqmm::tf_gather_db_live<T, G, OT, TFM>(
            x, xs, (const device uint8_t*)w, scales, biases, desc, y_col, y, params[0], params[1], Ws, SB,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    } else if constexpr (TFM == 4) {
        omlx_gqmm::tf_gather_db_half<T, G, OT>(
            x, (const device uint8_t*)w, scales, biases, desc, y_col, y, params[0], params[1], (threadgroup half*)Ws,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    } else {
        omlx_gqmm::tf_gather_db<T, G, OT, TFM>(
            x, xs, (const device uint8_t*)w, scales, biases, desc, y_col, y, params[0], params[1], Ws, SB,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    }
"""

# fp32 sum of each row's 64 activations of each group, in order (bf16 -> fp32 is a 16-bit shift).
RS_SOURCE = r"""
    const int M = params[0];
    const int K = params[1];
    const int Kg = K / 64;
    const int g = int(thread_position_in_grid.x);
    const int r = int(thread_position_in_grid.y);
    if (g >= Kg || r >= M) {
        return;
    }
    const device uint4* p = (const device uint4*)(x + size_t(r) * K + g * 64);
    float s = 0.0f;
    for (int i = 0; i < 8; i++) {
        const uint4 v = p[i];
        s += as_type<float>(v.x << 16); s += as_type<float>(v.x & 0xffff0000u);
        s += as_type<float>(v.y << 16); s += as_type<float>(v.y & 0xffff0000u);
        s += as_type<float>(v.z << 16); s += as_type<float>(v.z & 0xffff0000u);
        s += as_type<float>(v.w << 16); s += as_type<float>(v.w & 0xffff0000u);
    }
    xs[size_t(r) * Kg + g] = s;
"""

_K = {}

def _kernels():
    if not _K:
        mlx_src = NX._read_mlx_headers(NX._MLX_MM_HEADERS)
        assert mlx_src is not None, "mlx headers not found"
        _K["mm"] = mx.fast.metal_kernel(name="omlx_gqmm_tf_codes", input_names=["x", "w", "scales", "biases", "xs", "tiles", "tile_count", "params"],
                                        output_names=["y"], header=mlx_src + NX._MM_HEADER + TF_HEADER, source=TF_SOURCE)
        _K["rs"] = mx.fast.metal_kernel(name="omlx_tf_rowsums", input_names=["x", "params"], output_names=["xs"], source=RS_SOURCE)
        _K["scan"] = NX._get_kernel("scan")
    return _K

def row_sums(x):
    """x [M, 1, K] bf16 -> [M, K / 64] fp32 group sums."""
    k = _kernels()
    M, K = int(x.shape[0]), int(x.shape[-1]); Kg = K // 64
    tx = min(Kg, 256); ty = max(1, 256 // tx)
    return k["rs"](inputs=[x, mx.array([M, K], dtype=mx.int32)], grid=(Kg, M, 1), threadgroup=(tx, ty, 1),
                   output_shapes=[(M, Kg)], output_dtypes=[mx.float32])[0]

def plan_for(M, E, K, N):
    """The stock plan's tile height and layout, on the db schedule (seg 128-row plans become db 128)."""
    p = NX._plan(M, E, K, N)
    bm = int(os.environ.get("TF_BM", "0")) or p.bm
    gx = p.gx if os.environ.get("TF_GX", "") == "" else int(os.environ["TF_GX"])
    return bm, gx

def tf_sorted_gather_qmm(x, w, scales, biases, idx, xs=None, out_dtype=None, bm=None, gx=None, tfm=None):
    """x [M, 1, K] bf16 sorted rows, w [E, N, K / 8] uint32 4-bit g64, idx [M] uint32 sorted -> [M, 1, N]."""
    k = _kernels()
    M, K = int(idx.shape[0]), int(x.shape[-1]); E, N = int(w.shape[0]), int(w.shape[1])
    assert K % 64 == 0 and N % 64 == 0 and x.dtype == mx.bfloat16 and scales.dtype == mx.bfloat16
    if bm is None or gx is None:
        pbm, pgx = plan_for(M, E, K, N); bm = bm or pbm; gx = pgx if gx is None else gx
    if xs is None: xs = row_sums(x)
    max_tiles = (M + bm - 1) // bm + min(E, M)
    tiles, tile_count = k["scan"](inputs=[idx, mx.array([M, E, max_tiles], dtype=mx.int32)], template=[("BM", bm), ("MAXE", NX._MAX_EXPERTS)],
                                  grid=(1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(max_tiles * 4,), (1,)], output_dtypes=[mx.uint32, mx.uint32])
    n_cols = N // 64
    tg = (gx, ((max_tiles + gx - 1) // gx) * n_cols) if gx else (n_cols, max_tiles)
    ot = out_dtype or x.dtype
    return k["mm"](inputs=[x, w, scales, biases, xs, tiles, tile_count, mx.array([N, K], dtype=mx.int32)],
                   template=[("T", x.dtype), ("OT", ot), ("BM", bm), ("GX", gx), ("TFM", int(os.environ.get("TFM", "0")) if tfm is None else tfm)],
                   grid=(tg[0] * 32, tg[1] * 2, bm // 32), threadgroup=(32, 2, bm // 32),
                   output_shapes=[(M, 1, N)], output_dtypes=[ot])[0]

def stock(x, w, scales, biases, idx):
    return NX.sorted_gather_qmm(x, w, scales, biases, idx, group_size=64, bits=4, mode="affine", stream=None)

# ---------------------------------------------------------------------------------------------------------------
E, K, NH = 288, 4096, 1024

def qstack(n, k, scale=0.02, seed=2):
    mx.random.seed(seed)
    ws, ss, bs = [], [], []
    for i in range(0, E, 24):
        w = (mx.random.normal((24, n, k)) * scale).astype(mx.float32).reshape(24 * n, k)
        wq, sc, bi = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=4); mx.eval(wq, sc, bi)
        ws.append(wq.reshape(24, n, -1)); ss.append(sc.reshape(24, n, -1).astype(mx.bfloat16)); bs.append(bi.reshape(24, n, -1).astype(mx.bfloat16))
    out = [mx.concatenate(a, axis=0) for a in (ws, ss, bs)]; mx.eval(out); mx.clear_cache(); return out

def route(T, seed=0):
    mx.random.seed(100 + seed)
    r = mx.random.uniform(shape=(T, E)); top = mx.argpartition(-r, kth=7, axis=-1)[:, :8].reshape(-1)
    order = mx.argsort(top); idx = top[order].astype(mx.uint32); row_map = order // 8
    mx.eval(idx, row_map); return idx, row_map

def deq64(t, experts, rows=None):
    """float64 dequantized weights of the given experts (and output rows): exact s * q + b from the stored bf16."""
    wq, sc, bi = t
    out = {}
    for e in experts:
        q = np.array(wq[e]).view(np.uint32)
        codes = np.stack([(q >> (4 * j)) & 15 for j in range(8)], axis=-1).reshape(q.shape[0], -1).astype(np.float64)
        s = np.array(sc[e].astype(mx.float32)).astype(np.float64); b = np.array(bi[e].astype(mx.float32)).astype(np.float64)
        Wd = codes.reshape(q.shape[0], -1, 64) * s[:, :, None] + b[:, :, None]
        out[e] = Wd.reshape(q.shape[0], -1)
    return out

def selftest():
    # Canary: every plan height, partial tiles, an empty expert, K = 256 and 4096, N = 128.
    counts = (70, 0, 5, 33, 64, 17, 140, 11, 1, 2, 96, 3)
    Ec = len(counts)
    for Kc, Nc in ((256, 128), (4096, 128), (1024, 256)):
        mx.random.seed(7)
        wf = (mx.random.normal((Ec, Nc, Kc)) * 0.05).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(wf, group_size=64, bits=4)
        wd = mx.dequantize(wq, sc, bi, group_size=64, bits=4).astype(mx.float32)
        idx = mx.array([e for e, n in enumerate(counts) for _ in range(n)], dtype=mx.uint32)
        M = int(idx.shape[0])
        x = (mx.random.normal((M, 1, Kc)) * 0.5).astype(mx.bfloat16)
        ref = (x.astype(mx.float32) @ mx.take(wd, idx, axis=0).swapaxes(-1, -2))
        st = stock(x, wq, sc, bi, idx)
        for bm, gx, tfm in [(bm, gx, t) for t in (5, 6) for bm in (64, 96, 128) for gx in (0, 32)]:
                y = tf_sorted_gather_qmm(x, wq, sc, bi, idx, bm=bm, gx=gx, tfm=tfm)
                y32 = tf_sorted_gather_qmm(x, wq, sc, bi, idx, bm=bm, gx=gx, out_dtype=mx.float32, tfm=tfm)
                mx.eval(y, y32, ref, st)
                scale = float(mx.abs(ref).max().item())
                e_new = float(mx.abs(y.astype(mx.float32) - ref).max().item())
                e_32 = float(mx.abs(y32 - ref).max().item())
                e_st = float(mx.abs(st.astype(mx.float32) - ref).max().item()) if st is not None else float("nan")
                ok = e_new <= scale / 64 and math.isfinite(e_new)
                print(f"selftest K={Kc} N={Nc} bm={bm} gx={gx} tfm={tfm}: max|ref| {scale:.3f}  max err bf16 out {e_new:.3e}  fp32 out {e_32:.3e}  stock {e_st:.3e}  {'OK' if ok else 'FAIL'}", flush=True)

def compile_check():
    # tiny problem (a few KB, microseconds of GPU): only builds every variant
    mx.random.seed(1)
    wf = (mx.random.normal((2, 64, 128)) * 0.05).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(wf, group_size=64, bits=4)
    idx = mx.array([0] * 9 + [1] * 7, dtype=mx.uint32); x = (mx.random.normal((16, 1, 128)) * 0.5).astype(mx.bfloat16)
    for tfm in [int(v) for v in os.environ.get("TFMS", "0,1,2,3,4").split(",")]:
        for bm in (64, 96, 128):
            y = tf_sorted_gather_qmm(x, wq, sc, bi, idx, bm=bm, gx=0, tfm=tfm); mx.eval(y)
        print("compiled tfm", tfm, flush=True)

def chain(fn, x, n):
    prev = None
    for _ in range(n): prev = fn(x if prev is None else mx.depends(x, [prev]))
    return prev

def bench(fn, x, n=6, reps=4):
    mx.eval(chain(fn, x, n)); best = 1e9
    for _ in range(reps):
        o = chain(fn, x, n); t = time.perf_counter(); mx.eval(o); best = min(best, (time.perf_counter() - t) / n)
    return best * 1e3

def speed():
    gate = qstack(NH, K); down = qstack(K, NH, seed=3)
    tfms = [int(v) for v in os.environ.get("TFMS", "0").split(",")]
    print(f"weights per Mac per layer: gate (= up) {sum(a.nbytes for a in gate) / 1e9:.2f} GB, down {sum(a.nbytes for a in down) / 1e9:.2f} GB; modes {tfms} (0 epilogue each step, 1 none (timing only), 2 deferred)", flush=True)
    print(f"{'tokens':>6} {'plan':>5} | stock ms: {'gate':>6} {'down':>6} {'layer':>6} {'x42':>6} | per mode: rsum, gate, down, layer, x42, gain")
    for T in [int(v) for v in os.environ.get("TS", "48,64,128,144,300,600,1024,2048").split(",")]:
        idx, row_map = route(T)
        x = (mx.random.normal((T, K)) * 0.35).astype(mx.bfloat16); xr = mx.contiguous(x[row_map][:, None, :])
        h = (mx.random.normal((T * 8, 1, NH)) * 0.1).astype(mx.bfloat16); mx.eval(xr, h)
        M = T * 8
        pg = NX._plan(M, E, K, NH)
        sg = bench(lambda xx: stock(xx, *gate, idx), xr); sd = bench(lambda hh: stock(hh, *down, idx), h)
        sl = 2 * sg + sd
        rs = bench(lambda xx: row_sums(xx), xr); rsd = bench(lambda hh: row_sums(hh), h)
        xs = row_sums(xr); hs = row_sums(h); mx.eval(xs, hs)
        line = f"{T:6d} {pg.bm:5d} | {'':9} {sg:6.3f} {sd:6.3f} {sl:6.3f} {42 * sl:6.1f} |"
        for m in tfms:
            ng = bench(lambda xx: tf_sorted_gather_qmm(xx, *gate, idx, xs=xs, tfm=m), xr)
            nd = bench(lambda hh: tf_sorted_gather_qmm(hh, *down, idx, xs=hs, tfm=m), h)
            r_ = 0.0 if m in (1, 4) else rs + rsd   # modes 1 and 4 read no group sums
            nl = 2 * ng + nd + r_
            line += f" m{m}: {r_:.3f} {ng:.3f} {nd:.3f} {nl:.3f} {42 * nl:.1f} {100 * (1 - nl / sl):+.1f}% |"
        print(line, flush=True)

def acc():
    gate = qstack(NH, K); down = qstack(K, NH, seed=3)
    tfms = [int(v) for v in os.environ.get("TFMS", "0").split(",")]
    rng = np.random.default_rng(5)
    print(f"{'proj':>5} {'tokens':>6} {'mode':>5} | rel RMS err vs float64: stock(bf16 out) new(bf16 out) new(fp32 out) | max|stock-new| / max|ref| | bit-identical")
    for T in [int(v) for v in os.environ.get("TS", "64,600,2048").split(",")]:
        idx, row_map = route(T, seed=T)
        x = (mx.random.normal((T, K)) * 0.35).astype(mx.bfloat16); xr = mx.contiguous(x[row_map][:, None, :])
        # down input: SwiGLU-like (heavy-tailed, small)
        g = mx.random.normal((T * 8, 1, NH)); u = mx.random.normal((T * 8, 1, NH))
        h = ((g * mx.sigmoid(g)) * u * 0.15).astype(mx.bfloat16); mx.eval(xr, h)
        for name, t, xin in (("gate", gate, xr), ("down", down, h)):
            st = stock(xin, *t, idx); mx.eval(st)
            M = int(idx.shape[0]); sel = np.sort(rng.choice(M, min(M, int(os.environ.get("ROWS", "32"))), replace=False))
            ii = np.array(idx)[sel]
            X = np.array(xin.astype(mx.float32))[sel, 0].astype(np.float64)
            ref = []
            for j, e in enumerate(ii):           # one expert's float64 weights at a time (34 MB)
                ref.append(deq64(t, [int(e)])[int(e)] @ X[j])
            ref = np.stack(ref)
            A = np.array(st.astype(mx.float32))[sel, 0].astype(np.float64)
            rms = lambda z: float(np.sqrt(np.mean(z ** 2)))
            for m in tfms:
                nw = tf_sorted_gather_qmm(xin, *t, idx, tfm=m); n32 = tf_sorted_gather_qmm(xin, *t, idx, out_dtype=mx.float32, tfm=m)
                mx.eval(nw, n32)
                B = np.array(nw.astype(mx.float32))[sel, 0].astype(np.float64); C = np.array(n32)[sel, 0].astype(np.float64)
                same = bool(mx.array_equal(st, nw).item())
                print(f"{name:>5} {T:6d} {m:5d} | {rms(A - ref) / rms(ref):38.3e} {rms(B - ref) / rms(ref):13.3e} {rms(C - ref) / rms(ref):13.3e} | {np.abs(A - B).max() / np.abs(ref).max():24.3e} | {same}", flush=True)

if __name__ == "__main__":
    mode = os.environ.get("MODE", "selftest")
    print("mlx", mx.__version__, "mode", mode, flush=True)
    {"selftest": selftest, "speed": speed, "acc": acc, "compile": compile_check}[mode]()
