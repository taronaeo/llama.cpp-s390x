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
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

void ggml_gemm_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__VXE__) || defined(__VXE2__)
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);

            float32x4_t sumf[4];
            for (int m = 0; m < 4; m++) {
                sumf[m] = vec_splats(0.0f);
            }

            for (int l = 0; l < nb; l++) {
                float a_df[] = {
                    GGML_CPU_FP16_TO_FP32(a_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(a_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(a_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(a_ptr[l].d[3]),
                };
                float b_df[] = {
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_TO_FP32(b_ptr[l].d[3]),
                };

                float32x4_t a_d = vec_xl(0, (const float *)a_df);
                float32x4_t b_d = vec_xl(0, (const float *)b_df);

                int32x4_t sumi_0 = vec_splats(0);
                int32x4_t sumi_1 = vec_splats(0);
                int32x4_t sumi_2 = vec_splats(0);
                int32x4_t sumi_3 = vec_splats(0);

                for (int k_group = 0; k_group < 8; k_group += 4) {
                    ggml_int8x16x4_t a = ggml_vec_xl_s8x4(a_ptr[l].qs + 16 * k_group);
                    ggml_int8x16x4_t b = ggml_vec_xl_s8x4(b_ptr[l].qs + 16 * k_group);

                    for (int k = 0; k < 4; k++) {
                        sumi_0 = ggml_vdot_laneq_s32(sumi_0, b.val[k], a.val[k], 0);
                        sumi_1 = ggml_vdot_laneq_s32(sumi_1, b.val[k], a.val[k], 1);
                        sumi_2 = ggml_vdot_laneq_s32(sumi_2, b.val[k], a.val[k], 2);
                        sumi_3 = ggml_vdot_laneq_s32(sumi_3, b.val[k], a.val[k], 3);
                    }
                }

                sumf[0] = vec_madd(ggml_vmulq_laneq_f32(b_d, a_d, 0), vec_float(sumi_0), sumf[0]);
                sumf[1] = vec_madd(ggml_vmulq_laneq_f32(b_d, a_d, 1), vec_float(sumi_1), sumf[1]);
                sumf[2] = vec_madd(ggml_vmulq_laneq_f32(b_d, a_d, 2), vec_float(sumi_2), sumf[2]);
                sumf[3] = vec_madd(ggml_vmulq_laneq_f32(b_d, a_d, 3), vec_float(sumi_3), sumf[3]);
            }

            for (int m = 0; m < 4; m++) {
                vec_xst(sumf[m], 0, s + (y * 4 + m) * bs + x * 4);
            }
        }
    }
    return;
#endif  // defined(__VXE__) || defined(__VXE2__)
    ggml_gemm_q8_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__VXE__) || defined(__VXE2__)
    const block_q8_0x4 * b_ptr_base = (const block_q8_0x4 *) vx;

    for (int y = 0; y < nr; y += 4) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) vy + (y / 4) * nb;

        for (int x = 0; x < nc; x += ncols_interleaved) {
            const block_q8_0x4 * b_ptr = b_ptr_base + (x / 4) * nb;
            const block_q8_0x4 * a_ptr = a_ptr_base;

            float32x4_t acc_f32[4];
            for (int i = 0; i < 4; i++) {
                acc_f32[i] = vec_splats(0);
            }

            for (int b = 0; b < nb; b++) {
                int32x4_t acc[4];
                for (int i = 0; i < 4; i++) {
                    acc[i] = vec_splats(0);
                }

                // Process 4 chunks of 8 positions each
                for (int chunk = 0; chunk < 4; chunk++) {
                    int8x16_t a01 = vec_xl(0, a_ptr->qs + chunk * 32);
                    int8x16_t a23 = vec_xl(0, a_ptr->qs + chunk * 32 + 16);
                    int8x16_t b01 = vec_xl(0, b_ptr->qs + chunk * 32);
                    int8x16_t b23 = vec_xl(0, b_ptr->qs + chunk * 32 + 16);

                    acc[0] = vmmlaq_s32(acc[0], a01, b01);
                    acc[1] = vmmlaq_s32(acc[1], a01, b23);
                    acc[2] = vmmlaq_s32(acc[2], a23, b01);
                    acc[3] = vmmlaq_s32(acc[3], a23, b23);
                }

                // Reorder outputs from 2×2 tiles to row-major
                // acc[0] = [r0c0, r0c1, r1c0, r1c1]
                // acc[1] = [r0c2, r0c3, r1c2, r1c3]
                // acc[2] = [r2c0, r2c1, r3c0, r3c1]
                // acc[3] = [r2c2, r2c3, r3c2, r3c3]
                int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
                int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
                int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
                int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));

                // Scales
                float32x4_t a_d = vcvt_f32_f16(vld1_f16((const __fp16 *) a_ptr->d));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const __fp16 *) b_ptr->d));

                acc_f32[0] = vfmaq_f32(acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
                acc_f32[1] = vfmaq_f32(acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
                acc_f32[2] = vfmaq_f32(acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
                acc_f32[3] = vfmaq_f32(acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));

                a_ptr++;
                b_ptr++;
            }

            for (int row = 0; row < 4; row++) {
                vec_xst(acc_f32[row], 0, s + (y + row) * bs + x);
            }
        }
    }
    return;
#endif  // defined(__VXE__) || defined(__VXE2__)
    ggml_gemm_q8_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
