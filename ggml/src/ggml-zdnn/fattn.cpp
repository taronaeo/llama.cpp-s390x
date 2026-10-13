#include "ggml.h"
#include "fattn.hpp"

#include <algorithm>
#include <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
static inline float ggml_zdnn_fattn_mask_value(ggml_fp16_t m) {
    return std::max(GGML_FP16_TO_FP32(m), -65504.0f);
}

// converted mask, reused while its rows and shape are unchanged, as every layer of a batch uses the same mask
static const zdnn_ztensor & ggml_zdnn_fattn_mask(ggml_backend_zdnn_context * ctx,
                                           const zdnn_tensor_desc          * pre_tfm_desc,
                                           const ggml_tensor               * mask) {

    const int64_t n_kv     = pre_tfm_desc->dim1;
    const int64_t n_rows   = pre_tfm_desc->layout == ZDNN_3DS ? pre_tfm_desc->dim2 : 1;
    const int64_t n_stacks = pre_tfm_desc->layout == ZDNN_3DS ? pre_tfm_desc->dim3 : pre_tfm_desc->dim2;

    std::unique_ptr<ggml_backend_zdnn_buffer> & cached = ctx->fattn_mask;
    std::vector<ggml_fp16_t>                  & rows   = ctx->fattn_mask_rows;

    const bool same_shape = cached != nullptr &&
                            cached->pre_tfm_desc.layout == pre_tfm_desc->layout &&
                            cached->pre_tfm_desc.dim4   == pre_tfm_desc->dim4 &&
                            cached->pre_tfm_desc.dim3   == pre_tfm_desc->dim3 &&
                            cached->pre_tfm_desc.dim2   == pre_tfm_desc->dim2 &&
                            cached->pre_tfm_desc.dim1   == pre_tfm_desc->dim1;

    bool same_rows = same_shape;
    for (int64_t ir = 0; same_rows && ir < n_rows; ir++) {
        same_rows = memcmp(rows.data() + ir*n_kv, (const char *) mask->data + ir*mask->nb[1], n_kv*sizeof(ggml_fp16_t)) == 0;
    }

    if (same_rows) {
        return cached->ztensor;
    }

    if (same_shape) {
        zdnn_reset_ztensor(&cached->ztensor);
    } else {
        if (cached != nullptr) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&cached->ztensor));
        }

        cached = std::make_unique<ggml_backend_zdnn_buffer>();
        cached->pre_tfm_desc = *pre_tfm_desc;
        ZDNN_CHECK(zdnn_generate_transformed_desc(&cached->pre_tfm_desc, &cached->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&cached->pre_tfm_desc, &cached->tfm_desc, &cached->ztensor));
    }

    ggml_zdnn_pool_alloc<float> mask_host_alloc(ctx->pool, n_stacks*n_rows*n_kv);
    float * mask_host = mask_host_alloc.get();

    rows.resize(n_rows*n_kv);

    ggml_zdnn_parallel_for(ctx->n_threads, n_rows, [&](int64_t ir_start, int64_t ir_end) {
        for (int64_t ir = ir_start; ir < ir_end; ir++) {
            const ggml_fp16_t * mask_row = (const ggml_fp16_t *)((const char *) mask->data + ir*mask->nb[1]);
            float             * host_row = mask_host + ir*n_kv;

            memcpy(rows.data() + ir*n_kv, mask_row, n_kv*sizeof(ggml_fp16_t));
            for (int64_t ik = 0; ik < n_kv; ik++) {
                host_row[ik] = ggml_zdnn_fattn_mask_value(mask_row[ik]);
            }
            for (int64_t is = 1; is < n_stacks; is++) {
                memcpy(mask_host + (is*n_rows + ir)*n_kv, host_row, n_kv*sizeof(float));
            }
        }
    });

    ZDNN_CHECK(zdnn_transform_ztensor(&cached->ztensor, mask_host));
    return cached->ztensor;
}

bool ggml_zdnn_supports_flash_attn_ext(const ggml_backend_zdnn_device_context * ctx_dev, const ggml_tensor * op) {
    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * k    = op->src[1];
    const ggml_tensor * v    = op->src[2];
    const ggml_tensor * mask = op->src[3];

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) op->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) op->op_params + 2, sizeof(float));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    if (op->src[4] != nullptr || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        return false;
    }

    if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }

    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1)) {
        return false;
    }

    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        return false;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t)) {
        return false;
    }

    const int64_t max_dim = ctx_dev->max_size;
    const int64_t group   = q->ne[2] / k->ne[2];

    return q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && group * q->ne[1] <= max_dim;
}

void ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t d_k       = q->ne[0];
    const int64_t n_q       = q->ne[1];
    const int64_t n_head    = q->ne[2];
    const int64_t n_kv      = k->ne[1];
    const int64_t n_head_kv = k->ne[2];
    const int64_t d_v       = v->ne[0];
    const int64_t group     = n_head / n_head_kv;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int32_t n_threads = n_q*n_kv >= 65536 ? ctx->n_threads : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    const bool single = n_q == 1;

    ggml_zdnn_pool_alloc<float>       q_host_alloc(ctx->pool, n_head*n_q*d_k);
    ggml_zdnn_pool_alloc<float>       o_host_alloc(ctx->pool, n_head*n_q*d_v);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_head_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_head_kv*n_kv*d_v);

    float       * q_host = q_host_alloc.get();
    float       * o_host = o_host_alloc.get();
    ggml_fp16_t * k_host = k_host_alloc.get();
    ggml_fp16_t * v_host = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_threads, n_head, [&](int64_t h_start, int64_t h_end) {
        for (int64_t h = h_start; h < h_end; h++) {
            for (int64_t iq = 0; iq < n_q; iq++) {
                const float * src = (const float *)((const char *) q->data + iq*q->nb[1] + h*q->nb[2]);
                float       * row = q_host + (h*n_q + iq)*d_k;
                for (int64_t i0 = 0; i0 < d_k; i0++) {
                    row[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_threads, n_head_kv*n_kv, [&](int64_t r_start, int64_t r_end) {
        for (int64_t r = r_start; r < r_end; r++) {
            const int64_t hk = r / n_kv;
            const int64_t ik = r % n_kv;
            memcpy(k_host + r*d_k, (const char *) k->data + ik*k->nb[1] + hk*k->nb[2], d_k*sizeof(ggml_fp16_t));
            memcpy(v_host + r*d_v, (const char *) v->data + ik*v->nb[1] + hk*v->nb[2], d_v*sizeof(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    if (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_head_kv, group, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_head_kv, group, d_v);
    } else {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_head, n_q, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_head, n_q, d_v);
    }
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &k_pre_tfm_desc, n_head_kv, n_kv, d_k);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &v_pre_tfm_desc, n_head_kv, n_kv, d_v);

    ZDNN_CHECK(zdnn_generate_transformed_desc(&q_pre_tfm_desc, &q_tfm_desc));
    ZDNN_CHECK(zdnn_generate_transformed_desc(&k_pre_tfm_desc, &k_tfm_desc));
    ZDNN_CHECK(zdnn_generate_transformed_desc(&v_pre_tfm_desc, &v_tfm_desc));
    ZDNN_CHECK(zdnn_generate_transformed_desc(&o_pre_tfm_desc, &o_tfm_desc));

    ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &q_pre_tfm_desc, &q_tfm_desc, &q_ztensor, q_ztensor_alloc);
    ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &k_pre_tfm_desc, &k_tfm_desc, &k_ztensor, k_ztensor_alloc);
    ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &v_pre_tfm_desc, &v_tfm_desc, &v_ztensor, v_ztensor_alloc);
    ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &o_pre_tfm_desc, &o_tfm_desc, &o_ztensor, o_ztensor_alloc);

    ZDNN_CHECK(zdnn_transform_ztensor(&q_ztensor, q_host));
    ZDNN_CHECK(zdnn_transform_ztensor(&k_ztensor, k_host));
    ZDNN_CHECK(zdnn_transform_ztensor(&v_ztensor, v_host));

    ggml_zdnn_pool_alloc<uint8_t> softmax_save_area(ctx->pool, ZDNN_SOFTMAX_SAVEAREA_SIZE);

    zdnn_tensor_desc s_pre_tfm_desc, s_tfm_desc;
    zdnn_ztensor     s_ztensor, s2_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> s_ztensor_alloc, s2_ztensor_alloc;

    if (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        const zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_head_kv, n_kv);
        if (mask != nullptr) {
            zdnn_tensor_desc bias_pre_tfm_desc;
            zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias_pre_tfm_desc, n_head_kv, n_kv);
            bias = &ggml_zdnn_fattn_mask(ctx, &bias_pre_tfm_desc, mask);
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, n_head_kv, group, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, false, true, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_head_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } else {
        const zdnn_ztensor * mask_ztensor = nullptr;
        if (mask != nullptr) {
            zdnn_tensor_desc mask_pre_tfm_desc;
            zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &mask_pre_tfm_desc, 1, n_q, n_kv);
            mask_ztensor = &ggml_zdnn_fattn_mask(ctx, &mask_pre_tfm_desc, mask);
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, 1, n_q, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        const zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        const zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        for (int64_t h = 0; h < n_head; h++) {
            const zdnn_ztensor q_view = ggml_zdnn_stack_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_head,    h,         1);
            const zdnn_ztensor k_view = ggml_zdnn_stack_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_head_kv, h / group, 1);
            const zdnn_ztensor v_view = ggml_zdnn_stack_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_head_kv, h / group, 1);
            zdnn_ztensor       o_view = ggml_zdnn_stack_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_head,    h,         1);

            s_ztensor.is_transformed  = false;
            s2_ztensor.is_transformed = false;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, false, true, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            if (mask != nullptr) {
                ZDNN_CHECK(zdnn_add(&s_ztensor, mask_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = false;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = true;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    for (int64_t iq = 0; iq < n_q; iq++) {
        for (int64_t h = 0; h < n_head; h++) {
            memcpy((char *) dst->data + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*sizeof(float));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_buffer * dst_extra = (ggml_backend_zdnn_buffer *) dst->extra;
    if (dst_extra != nullptr) {
        dst_extra->ztensor.is_transformed = false;
    }
}
