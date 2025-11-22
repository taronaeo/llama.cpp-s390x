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

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float32x4_t acc = vec_splats(0.0f);

        for (int b = 0; b < nb; b++) {
            int8x16_t  b0 = vec_xl(0, (const int8_t *) b_ptr->qs +  0);
            int8x16_t  b1 = vec_xl(0, (const int8_t *) b_ptr->qs + 16);
            int8x16_t  b2 = vec_xl(0, (const int8_t *) b_ptr->qs + 32);
            int8x16_t  b3 = vec_xl(0, (const int8_t *) b_ptr->qs + 48);
            uint16x8_t bd = vec_xl(0, (const uint16_t *) b_ptr->d);

            int8x16_t  a0 = vec_xl(0, a_ptr->qs);
            int8x16_t  a1 = vec_xl(0, a_ptr->qs + qk/2);
            uint16x8_t ad = vec_xl(0, (const uint16_t *) &a_ptr->d);

            int32x4_t ret = vec_splats(0);

            ret = vec_madd_lane_s8(ret, b0 << 4, a0, 0);
            ret = vec_madd_lane_s8(ret, b1 << 4, a0, 1);
            ret = vec_madd_lane_s8(ret, b2 << 4, a0, 2);
            ret = vec_madd_lane_s8(ret, b3 << 4, a0, 3);

            ret = vec_madd_lane_s8(ret, b0 & 0xf0U, a1, 0);
            ret = vec_madd_lane_s8(ret, b1 & 0xf0U, a1, 1);
            ret = vec_madd_lane_s8(ret, b2 & 0xf0U, a1, 2);
            ret = vec_madd_lane_s8(ret, b3 & 0xf0U, a1, 3);

            int32x4_t adf = { GGML_COMPUTE_FP16_TO_FP32(ad[0]), GGML_COMPUTE_FP16_TO_FP32(ad[1]), GGML_COMPUTE_FP16_TO_FP32(ad[2]), GGML_COMPUTE_FP16_TO_FP32(ad[3]) };
            int32x4_t bdf = { GGML_COMPUTE_FP16_TO_FP32(bd[0]), GGML_COMPUTE_FP16_TO_FP32(bd[1]), GGML_COMPUTE_FP16_TO_FP32(bd[2]), GGML_COMPUTE_FP16_TO_FP32(bd[3]) };

            acc = vec_madd(
                vec_float(ret),
                vec_mul(
                    vec_float(adf),
                    vec_float(bdf)
                ),
            acc);

            a_ptr++;
            b_ptr++;
        }

        vec_xst(acc, 0, (float *)(s));
        s += ncols_interleaved;
    }
    return;
#endif
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
