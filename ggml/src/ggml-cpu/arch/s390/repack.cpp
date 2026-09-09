#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cstdio>

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#define UNUSED GGML_UNUSED

#if defined(__VXE__) || defined(__VXE2__)

static inline int8x16_t ggml_vxe_q1_0_signs(const uint8x16_t v_x, const uint8x16_t v_bit) {
    // isolate each lane's bit, then set all ones where that bit is clear, the -d case
    return (int8x16_t)vec_cmpeq(vec_and(v_x, v_bit), vec_splats((uint8_t)0x00));
}

static inline uint32x4_t ggml_vxe_q1_0_acc(const uint32x4_t v_p, const int8x16_t v_y, const int8x16_t v_m) {
    const int8x16_t v_ys = vec_sub(vec_xor(v_y, v_m), v_m);
    return vec_add(v_p, vec_sum4(vec_xor((uint8x16_t)v_ys, vec_splats((uint8_t)0x80)), vec_splats((uint8_t)0x00)));
}

static inline int32x4_t ggml_vxe_q1_0_dot(const int8x16_t v_ml, const int8x16_t v_mh,
                                          const int8x16_t v_yl, const int8x16_t v_yh) {
    const uint8x16_t v_zero = vec_splats((uint8_t)0x00);
    const uint8x16_t v_bias = vec_splats((uint8_t)0x80);  // v ^ 0x80 == v + 128

    // weights are only +1 or -1, so negate y
    const int8x16_t v_ysl = vec_sub(vec_xor(v_yl, v_ml), v_ml);
    const int8x16_t v_ysh = vec_sub(vec_xor(v_yh, v_mh), v_mh);

    // bias to unsigned, then vec_sum4 adds each group of 4 bytes into one word
    const uint32x4_t v_p = vec_add(vec_sum4(vec_xor((uint8x16_t)v_ysl, v_bias), v_zero),
                                   vec_sum4(vec_xor((uint8x16_t)v_ysh, v_bias), v_zero));

    // each word summed 8 biased bytes, so take back 8 * 128
    return vec_sub((int32x4_t)v_p, vec_splats((int32_t)1024));
}

#endif

void ggml_gemv_q1_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK1_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);
    UNUSED(qk);
    UNUSED(nb);
    UNUSED(ncols_interleaved);

#if defined(__VXE__) || defined(__VXE2__)
    // one qs byte holds 4 weights of one row per nibble, so two broadcasts feed all 4 rows
    const uint8x16_t v_idxl = { 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6 };
    const uint8x16_t v_idxh = vec_add(v_idxl, vec_splats((uint8_t)8));
    const uint8x16_t v_one  = vec_splats((uint8_t)1);

    const uint8x16_t v_bitl = { 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8 };
    const uint8x16_t v_bith = vec_sl(v_bitl, 4);

    const block_q8_0 * GGML_RESTRICT a_ptr = (const block_q8_0 *)vy;

    for (int x = 0; x < nc / ncols_interleaved; ++x) {
        const block_q1_0x4 * GGML_RESTRICT b_ptr = (const block_q1_0x4 *)vx + (x * nb);

        float32x4_t v_sumf0 = vec_splats(0.0f);
        float32x4_t v_sumf1 = vec_splats(0.0f);
        float32x4_t v_sumf2 = vec_splats(0.0f);
        float32x4_t v_sumf3 = vec_splats(0.0f);

        for (int l = 0; l < nb; ++l) {
            const float32x4_t v_xd0 = vec_splats(GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]));
            const float32x4_t v_xd1 = vec_splats(GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]));
            const float32x4_t v_xd2 = vec_splats(GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]));
            const float32x4_t v_xd3 = vec_splats(GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]));

            for (int k = 0; k < QK1_0 / QK8_0; ++k) {
                const block_q8_0 * GGML_RESTRICT yb = a_ptr + l * (QK1_0 / QK8_0) + k;
                const float32x4_t v_yd = vec_splats(GGML_CPU_FP16_TO_FP32(yb->d));

                const uint8x16_t v_x = vec_xl(k * 16, (const uint8_t *)b_ptr[l].qs);

                const int8x16_t v_yl = vec_xl(0,         (const int8_t *)yb->qs);
                const int8x16_t v_yh = vec_xl(QK8_0 / 2, (const int8_t *)yb->qs);

                // even qs bytes carry rows 0 and 1, odd bytes carry rows 2 and 3
                const uint8x16_t v_xel = vec_perm(v_x, v_x, v_idxl);
                const uint8x16_t v_xeh = vec_perm(v_x, v_x, v_idxh);
                const uint8x16_t v_xol = vec_perm(v_x, v_x, vec_add(v_idxl, v_one));
                const uint8x16_t v_xoh = vec_perm(v_x, v_x, vec_add(v_idxh, v_one));

                const int32x4_t v_xy0 = ggml_vxe_q1_0_dot(ggml_vxe_q1_0_signs(v_xel, v_bitl),
                                                          ggml_vxe_q1_0_signs(v_xeh, v_bitl), v_yl, v_yh);
                const int32x4_t v_xy1 = ggml_vxe_q1_0_dot(ggml_vxe_q1_0_signs(v_xel, v_bith),
                                                          ggml_vxe_q1_0_signs(v_xeh, v_bith), v_yl, v_yh);
                const int32x4_t v_xy2 = ggml_vxe_q1_0_dot(ggml_vxe_q1_0_signs(v_xol, v_bitl),
                                                          ggml_vxe_q1_0_signs(v_xoh, v_bitl), v_yl, v_yh);
                const int32x4_t v_xy3 = ggml_vxe_q1_0_dot(ggml_vxe_q1_0_signs(v_xol, v_bith),
                                                          ggml_vxe_q1_0_signs(v_xoh, v_bith), v_yl, v_yh);

                v_sumf0 = vec_madd(vec_float(v_xy0), vec_mul(v_xd0, v_yd), v_sumf0);
                v_sumf1 = vec_madd(vec_float(v_xy1), vec_mul(v_xd1, v_yd), v_sumf1);
                v_sumf2 = vec_madd(vec_float(v_xy2), vec_mul(v_xd2, v_yd), v_sumf2);
                v_sumf3 = vec_madd(vec_float(v_xy3), vec_mul(v_xd3, v_yd), v_sumf3);
            }
        }

        s[x * ncols_interleaved + 0] = vec_hsum_f32x4(v_sumf0);
        s[x * ncols_interleaved + 1] = vec_hsum_f32x4(v_sumf1);
        s[x * ncols_interleaved + 2] = vec_hsum_f32x4(v_sumf2);
        s[x * ncols_interleaved + 3] = vec_hsum_f32x4(v_sumf3);
    }

    return;
#endif
    ggml_gemv_q1_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q1_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK1_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(qk);
    UNUSED(nb);
    UNUSED(ncols_interleaved);

#if defined(__VXE__) || defined(__VXE2__)
    // a qs byte holds 4 weights of one row per nibble, and the sign of lane L
    // depends only on L % 4, so one broadcast byte drives all 16 lanes
    const uint8x16_t v_bitl = { 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8 };
    const uint8x16_t v_bith = vec_sl(v_bitl, 4);

    for (int y = 0; y < nr / 4; ++y) {
        const block_q8_0x4 * GGML_RESTRICT a_ptr = (const block_q8_0x4 *)vy + (4 * y * nb);

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q1_0x4 * GGML_RESTRICT b_ptr = (const block_q1_0x4 *)vx + (x * nb);

            // lane m of each accumulator holds the sum for activation row m
            float32x4_t v_acc0 = vec_splats(0.0f);
            float32x4_t v_acc1 = vec_splats(0.0f);
            float32x4_t v_acc2 = vec_splats(0.0f);
            float32x4_t v_acc3 = vec_splats(0.0f);

            for (int l = 0; l < nb; ++l) {
                for (int k = 0; k < QK1_0 / QK8_0; ++k) {
                    const block_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k;

                    uint32x4_t v_p0 = vec_splats((uint32_t)0);
                    uint32x4_t v_p1 = vec_splats((uint32_t)0);
                    uint32x4_t v_p2 = vec_splats((uint32_t)0);
                    uint32x4_t v_p3 = vec_splats((uint32_t)0);

                    for (int tile = 0; tile < QK8_0 / 4; ++tile) {
                        // 4 bytes per activation row, so vec_sum4 folds one partial per row
                        const int8x16_t v_y = vec_xl(tile * 16, (const int8_t *)a_blk->qs);

                        const uint8x16_t v_xl = vec_splats((uint8_t)b_ptr[l].qs[k * 16 + 2 * tile + 0]);
                        const uint8x16_t v_xh = vec_splats((uint8_t)b_ptr[l].qs[k * 16 + 2 * tile + 1]);

                        v_p0 = ggml_vxe_q1_0_acc(v_p0, v_y, ggml_vxe_q1_0_signs(v_xl, v_bitl));
                        v_p1 = ggml_vxe_q1_0_acc(v_p1, v_y, ggml_vxe_q1_0_signs(v_xl, v_bith));
                        v_p2 = ggml_vxe_q1_0_acc(v_p2, v_y, ggml_vxe_q1_0_signs(v_xh, v_bitl));
                        v_p3 = ggml_vxe_q1_0_acc(v_p3, v_y, ggml_vxe_q1_0_signs(v_xh, v_bith));
                    }

                    const float32x4_t v_ad = {
                        GGML_CPU_FP16_TO_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_TO_FP32(a_blk->d[3]),
                    };

                    // each word absorbed 8 tiles of 4 biased bytes, so take back 32 * 128
                    const int32x4_t v_bias32 = vec_splats((int32_t)4096);

                    v_acc0 = vec_madd(vec_float(vec_sub((int32x4_t)v_p0, v_bias32)),
                                      vec_mul(v_ad, vec_splats(GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]))), v_acc0);
                    v_acc1 = vec_madd(vec_float(vec_sub((int32x4_t)v_p1, v_bias32)),
                                      vec_mul(v_ad, vec_splats(GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]))), v_acc1);
                    v_acc2 = vec_madd(vec_float(vec_sub((int32x4_t)v_p2, v_bias32)),
                                      vec_mul(v_ad, vec_splats(GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]))), v_acc2);
                    v_acc3 = vec_madd(vec_float(vec_sub((int32x4_t)v_p3, v_bias32)),
                                      vec_mul(v_ad, vec_splats(GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]))), v_acc3);
                }
            }

            float tmp[4][4];
            memcpy(&tmp[0][0], &v_acc0, sizeof(float32x4_t));
            memcpy(&tmp[1][0], &v_acc1, sizeof(float32x4_t));
            memcpy(&tmp[2][0], &v_acc2, sizeof(float32x4_t));
            memcpy(&tmp[3][0], &v_acc3, sizeof(float32x4_t));

            for (int m = 0; m < 4; ++m) {
                for (int j = 0; j < ncols_interleaved; ++j) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = tmp[j][m];
                }
            }
        }
    }

    return;
#endif
    ggml_gemm_q1_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
