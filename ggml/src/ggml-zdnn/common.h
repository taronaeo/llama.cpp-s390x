#pragma once

#ifndef ZDNN_COMMON_H
#define ZDNN_COMMON_H

#include "zdnn.h"
#include "../include/ggml.h"
#include "../include/ggml-zdnn.h"

#include <memory>

#define ZDNN_CHECK(stmt)                \
    do {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } while (0);

#define BCAST_SHAPE(src0, src1)                                         \
    int64_t bcast_##src0##_ne[GGML_MAX_DIMS * 2];                       \
    int64_t bcast_##src1##_ne[GGML_MAX_DIMS * 2];                       \
    size_t  bcast_##src0##_nb[GGML_MAX_DIMS * 2];                       \
    size_t  bcast_##src1##_nb[GGML_MAX_DIMS * 2];                       \
    int64_t bcast_dims = ggml_zdnn_get_bcast_shape(src0, src1,          \
                                                   bcast_##src0##_ne,   \
                                                   bcast_##src1##_ne,   \
                                                   bcast_##src0##_nb,   \
                                                   bcast_##src1##_nb);

#define BCAST_PARAM(tensor) bcast_##tensor##_ne, bcast_##tensor##_nb, bcast_dims

struct ggml_backend_zdnn_context {
  // std::unique_ptr<char[]> work_data;
  // size_t work_size = 0;
};

static bool ggml_zdnn_need_bcast(const ggml_tensor * t0,
                                 const ggml_tensor * t1);

int ggml_zdnn_get_bcast_shape(const ggml_tensor * src0,
                              const ggml_tensor * src1,
                                    int64_t     * bcast_src0_ne,
                                    int64_t     * bcast_src1_ne,
                                    size_t      * bcast_src0_nb,
                                    size_t      * bcast_src1_nb);

static zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

void ggml_zdnn_create_tensor(const ggml_tensor      * tensor,
                                   zdnn_tensor_desc & pre_tfm_desc,
                                   zdnn_tensor_desc & tfm_desc,
                                   zdnn_ztensor     & ztensor,
                                   int64_t          * ne,
                                   size_t           * nb,
                                   int64_t            dims);

void ggml_zdnn_load_tensor(const ggml_tensor  * tensor,
                                 zdnn_ztensor & ztensor);

void ggml_zdnn_op_add(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_sub(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_mul(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_div(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);

void ggml_zdnn_op_log                   (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_exp                   (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_sqrt                  (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_invsqrt               (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_relu                  (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_leaky_relu            (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_tanh                  (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_sigmoid               (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_softmax               (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_softmax_mask          (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_gelu                  (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_meanreduce2d          (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_batchnorm             (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_norm                  (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_moments               (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_layernorm             (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_reduce                (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_matmul                (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_matmul_bcast          (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_matmul_transpose      (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_matmul_quantized      (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);

void ggml_zdnn_op_lstm          (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_gru           (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_avgpool2d     (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_maxpool2d     (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);
void ggml_zdnn_op_conv2d        (ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);

#endif /* ZDNN_COMMON_H */
