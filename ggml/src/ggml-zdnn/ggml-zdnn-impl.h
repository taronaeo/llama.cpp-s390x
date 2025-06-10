#pragma once

#ifndef GGML_ZDNN_IMPL
#define GGML_ZDNN_IMPL

#include "zdnn.h"
#include "../include/ggml.h"
#include "../include/ggml-zdnn.h"

#include <memory>

#define ZDNN_CHECK(stmt)                \
    do {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } while (0);

struct ggml_backend_zdnn_context {
  // std::unique_ptr<char[]> work_data;
  // size_t work_size = 0;
};

// --------------------------------------------------------------------------
// zDNN Internal Helper Functions
// --------------------------------------------------------------------------
void zdnn_tensor_bcast(const struct ggml_tensor * src,
                       const struct ggml_tensor * dst,
                                           void * dst_data,
                                         size_t   element_size);

// --------------------------------------------------------------------------
// zDNN Interfacing API
// --------------------------------------------------------------------------
static zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

void ggml_zdnn_create_tensor(const ggml_tensor      * tensor,
                                   zdnn_tensor_desc & pre_tfm_desc,
                                   zdnn_tensor_desc & tfm_desc,
                                   zdnn_ztensor     & ztensor,
                             const ggml_tensor      * dst);

void ggml_zdnn_load_tensor(const ggml_tensor  * tensor,
                                 zdnn_ztensor & ztensor);

static bool ggml_zdnn_compute_forward(struct ggml_backend_zdnn_context & ctx,
                                      struct               ggml_tensor * dst);

template<zdnn_status (*zdnn_op)(const zdnn_ztensor *, const zdnn_ztensor *, zdnn_ztensor *)>
void ggml_zdnn_op_bin(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);

template<zdnn_status (*zdnn_op)(const zdnn_ztensor *, zdnn_ztensor *)>
void ggml_zdnn_op_unary(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);

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


// --------------------------------------------------------------------------
// Backend buffer type
// --------------------------------------------------------------------------


// --------------------------------------------------------------------------
// Backend buffer
// --------------------------------------------------------------------------


// --------------------------------------------------------------------------
// Backend (stream)
// --------------------------------------------------------------------------


#endif  // GGML_ZDNN_IMPL
