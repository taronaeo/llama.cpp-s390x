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

void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__VXE__) || defined(__VXE2__)
    const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;
    const int8x16_t v_shift_4 = vec_splats((signed char)4);
    const int8x16_t v_mask_f0 = vec_splats((signed char)0xF0);
    const float32x4_t v_inv_16 = vec_splats(1.0f / 16.0f);

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float32x4_t acc = vec_splats(0.0f);

        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vec_xl(0, (const int8_t *) b_ptr->qs +  0);
            int8x16_t b1 = vec_xl(0, (const int8_t *) b_ptr->qs + 16);
            int8x16_t b2 = vec_xl(0, (const int8_t *) b_ptr->qs + 32);
            int8x16_t b3 = vec_xl(0, (const int8_t *) b_ptr->qs + 48);

            float32x4_t bd = {
                ggml_fp16_to_fp32(b_ptr->d[0]),
                ggml_fp16_to_fp32(b_ptr->d[1]),
                ggml_fp16_to_fp32(b_ptr->d[2]),
                ggml_fp16_to_fp32(b_ptr->d[3])
            };

            int8x16_t a0 = vec_xl(0, (const int8_t *) a_ptr->qs);
            int8x16_t a1 = vec_xl(0, (const int8_t *) a_ptr->qs + qk/2);
            float32x4_t ad = vec_splats(ggml_fp16_to_fp32(a_ptr->d));

            int32x4_t ret = vec_splats(0);

            ret = vec_madd_lane_s8(ret, vec_slb(b0, v_shift_4), a0, 0);
            ret = vec_madd_lane_s8(ret, vec_slb(b1, v_shift_4), a0, 1);
            ret = vec_madd_lane_s8(ret, vec_slb(b2, v_shift_4), a0, 2);
            ret = vec_madd_lane_s8(ret, vec_slb(b3, v_shift_4), a0, 3);

            ret = vec_madd_lane_s8(ret, vec_and(b0, v_mask_f0), a1, 0);
            ret = vec_madd_lane_s8(ret, vec_and(b1, v_mask_f0), a1, 1);
            ret = vec_madd_lane_s8(ret, vec_and(b2, v_mask_f0), a1, 2);
            ret = vec_madd_lane_s8(ret, vec_and(b3, v_mask_f0), a1, 3);

            float32x4_t ret_f = vec_float(ret);
            float32x4_t scaled_ret = vec_mul(ret_f, v_inv_16);
            float32x4_t scale_factors = vec_mul(ad, bd);

            acc = vec_madd(scaled_ret, scale_factors, acc);

            a_ptr++;
            b_ptr++;
        }

        vec_xst(acc, 0, s);
        s += ncols_interleaved;
    }
    return;
#endif
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
