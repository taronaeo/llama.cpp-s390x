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

void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__VXE__) || defined(__VXE2__)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vec_xl(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vec_max(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vec_max(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vec_max(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vec_max(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            float32x4_t v = vec_mul(srcv[0][j], id[0]);
            int32x4_t vi = vec_signed(v);
            y[i].qs[16 * j + 0] = vec_extract(vi, 0);
            y[i].qs[16 * j + 1] = vec_extract(vi, 1);
            y[i].qs[16 * j + 2] = vec_extract(vi, 2);
            y[i].qs[16 * j + 3] = vec_extract(vi, 3);

            v = vec_mul(srcv[1][j], id[1]);
            vi = vec_signed(v);
            y[i].qs[16 * j + 4] = vec_extract(vi, 0);
            y[i].qs[16 * j + 5] = vec_extract(vi, 1);
            y[i].qs[16 * j + 6] = vec_extract(vi, 2);
            y[i].qs[16 * j + 7] = vec_extract(vi, 3);

            v = vec_mul(srcv[2][j], id[2]);
            vi = vec_signed(v);
            y[i].qs[16 * j + 8]  = vec_extract(vi, 0);
            y[i].qs[16 * j + 9]  = vec_extract(vi, 1);
            y[i].qs[16 * j + 10] = vec_extract(vi, 2);
            y[i].qs[16 * j + 11] = vec_extract(vi, 3);

            v = vec_mul(srcv[3][j], id[3]);
            vi = vec_signed(v);
            y[i].qs[16 * j + 12] = vec_extract(vi, 0);
            y[i].qs[16 * j + 13] = vec_extract(vi, 1);
            y[i].qs[16 * j + 14] = vec_extract(vi, 2);
            y[i].qs[16 * j + 15] = vec_extract(vi, 3);
        }
    }
#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
}
