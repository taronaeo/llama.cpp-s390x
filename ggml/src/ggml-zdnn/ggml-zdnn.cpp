#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-zdnn/zdnn.h"
#include "ggml-zdnn/ggml-zdnn-impl.h"

#include <string>
#include <memory>
#include <stdint.h>
#include <csignal>
#include <vecintrin.h>

// --------------------------------------------------------------------------
// zDNN Internal Helper Functions
// --------------------------------------------------------------------------
void zdnn_tensor_pack(      void    * dst_buffer,
                      const void    * src_buffer,
                      const int64_t * ne,
                      const size_t  * nb,
                            size_t    element_size) {
    const int64_t src_w = ne[0];
    const int64_t src_h = ne[1];
    const int64_t src_c = ne[2];
    const int64_t src_n = ne[3];

    const int64_t total_elements = src_w * src_h * src_c * src_n;
    const size_t packed_size_bytes = total_elements * element_size;

    const char * src_ptr = (const char *)src_buffer;
          char * dst_ptr = (      char *)dst_buffer;

    for (int64_t i = 0; i < total_elements; i++) {
        int64_t w = i % src_w;
        int64_t h = (i / src_w) % src_h;
        int64_t c = (i / (src_w * src_h)) % src_c;
        int64_t n = i / (src_w * src_h * src_c);

        size_t src_offset = w * nb[0]
                          + h * nb[1]
                          + c * nb[2]
                          + n * nb[3];

        memcpy(dst_ptr, src_ptr + src_offset, element_size);
        dst_ptr += element_size;
    }
}

void zdnn_tensor_bcast(const ggml_tensor * src,
                       const ggml_tensor * dst,
                                    void * dst_buffer,
                                  size_t   element_size) {
    const int64_t src_w = src->ne[0];
    const int64_t src_h = src->ne[1];
    const int64_t src_c = src->ne[2];
    const int64_t src_n = src->ne[3];

    const int64_t dst_w = dst->ne[0];
    const int64_t dst_h = dst->ne[1];
    const int64_t dst_c = dst->ne[2];
    const int64_t dst_n = dst->ne[3];

    const int64_t total_elements = dst_w * dst_h * dst_c * dst_n;

    const char * src_ptr = (const char *)src->data;
          char * dst_ptr = (      char *)dst_buffer;

    for (int64_t i = 0; i < total_elements; i++) {
        int64_t w = i % dst_w;
        int64_t h = (i / dst_w) % dst_h;
        int64_t c = (i / (dst_w * dst_h)) % dst_c;
        int64_t n = i / (dst_w * dst_h * dst_c);

        int64_t src_w_idx = (src_w == 1) ? 0 : (w % src_w);
        int64_t src_h_idx = (src_h == 1) ? 0 : (h % src_h);
        int64_t src_c_idx = (src_c == 1) ? 0 : c;
        int64_t src_n_idx = (src_n == 1) ? 0 : n;

        size_t src_offset = src_w_idx * src->nb[0]
                          + src_h_idx * src->nb[1]
                          + src_c_idx * src->nb[2]
                          + src_n_idx * src->nb[3];

        size_t dst_offset = i * element_size;

        memcpy(dst_ptr + dst_offset,
               src_ptr + src_offset,
               element_size);
    }
}

inline void zdnn_transpose_4x4(const float * src,
                                     float * dst,
                                   int64_t   src_stride,
                                   int64_t   dst_stride) {
    float32x4_t row0 = vec_xl(0, src + 0 * src_stride);
    float32x4_t row1 = vec_xl(0, src + 1 * src_stride);
    float32x4_t row2 = vec_xl(0, src + 2 * src_stride);
    float32x4_t row3 = vec_xl(0, src + 3 * src_stride);

    float32x4_t tmp0 = vec_mergeh(row0, row1);
    float32x4_t tmp1 = vec_mergel(row0, row1);
    float32x4_t tmp2 = vec_mergeh(row2, row3);
    float32x4_t tmp3 = vec_mergel(row2, row3);

    float32x4_t col0 = vec_mergeh(tmp0, tmp2);
    float32x4_t col1 = vec_mergel(tmp0, tmp2);
    float32x4_t col2 = vec_mergeh(tmp1, tmp3);
    float32x4_t col3 = vec_mergel(tmp1, tmp3);

    vec_xst(col0, 0, dst + 0 * dst_stride);
    vec_xst(col1, 0, dst + 1 * dst_stride);
    vec_xst(col2, 0, dst + 2 * dst_stride);
    vec_xst(col3, 0, dst + 3 * dst_stride);
}

inline void zdnn_transpose(const float   * src,
                                 float   * dst,
                           const int64_t   rows,
                           const int64_t   cols) {
    const int64_t block_size = 4;

    for (int64_t i = 0; i < rows - block_size + 1; i += block_size) {
        for (int64_t j = 0; j < cols - block_size + 1; j += block_size) {
            zdnn_transpose_4x4(src + i * cols + j,
                               dst + j * rows + i,
                               cols,
                               rows);
        }

        for (int64_t j = (cols / block_size) * block_size; j < cols; j++) {
            for (int64_t k = 0; k < block_size && i + k < rows; k++) {
                dst[(j * rows) + (i + k)] = src[(i + k) * cols + j];
            }
        }
    }

    for (int64_t i = (rows / block_size) * block_size; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            dst[(j * rows) + i] = src[i * cols + j];
        }
    }
}


// --------------------------------------------------------------------------
// zDNN Interfacing API
// --------------------------------------------------------------------------
inline zdnn_data_types ggml_zdnn_type_mapping(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return FP32;
        case GGML_TYPE_F16:
            return FP16;
        case GGML_TYPE_BF16:
            return BFLOAT;
        case GGML_TYPE_I8:
            return INT8;
        case GGML_TYPE_I32:
            return INT32;
        case GGML_TYPE_Q8_0:
            return INT8;
        default:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            break;
    }
}

inline void ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                                    zdnn_tensor_desc  & tfm_desc,
                                    zdnn_ztensor      & ztensor,
                              const ggml_tensor       * src,
                              const int64_t           * ne,
                              const zdnn_data_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

inline void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor,
                                          void * buffer) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, buffer));
}

template<zdnn_status (*zdnn_op)(const zdnn_ztensor *, const zdnn_ztensor *, zdnn_ztensor *)>
void ggml_zdnn_op_bin(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor) {
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    const struct ggml_tensor * dst  = tensor;

    zdnn_tensor_desc pre_tfm_desc_src0, tfm_desc_src0;
    zdnn_tensor_desc pre_tfm_desc_src1, tfm_desc_src1;
    zdnn_tensor_desc pre_tfm_desc_dst , tfm_desc_dst;

    zdnn_ztensor ztensor_src0;
    zdnn_ztensor ztensor_src1;
    zdnn_ztensor ztensor_dst;

    ggml_zdnn_create_tensor(pre_tfm_desc_src0, tfm_desc_src0, ztensor_src0, src0, dst->ne, ZDNN_NCHW);
    ggml_zdnn_create_tensor(pre_tfm_desc_src1, tfm_desc_src1, ztensor_src1, src1, dst->ne, ZDNN_NCHW);
    ggml_zdnn_create_tensor(pre_tfm_desc_dst , tfm_desc_dst , ztensor_dst , dst , dst->ne, ZDNN_NCHW);

    void * src0_contiguous = nullptr;
    void * src1_contiguous = nullptr;

    size_t element_size = ggml_element_size(dst);
    size_t dst_buffer_size = ggml_nelements(dst) * element_size;

    // void * src0_packed = ggml_aligned_malloc(dst_buffer_size);
    // void * src1_packed = ggml_aligned_malloc(dst_buffer_size);

    if (ggml_are_same_shape(src0, dst)) {
        src0_contiguous = (void *)src0->data;
    } else {
        src0_contiguous = ggml_aligned_malloc(dst_buffer_size);
        zdnn_tensor_bcast(src0, dst, src0_contiguous, element_size);
    }

    // zdnn_tensor_pack(src0_packed, src0_contiguous, dst->ne, dst->nb, element_size);

    if (ggml_are_same_shape(src1, dst)) {
        src1_contiguous = (void *)src1->data;
    } else {
        src1_contiguous = ggml_aligned_malloc(dst_buffer_size);
        zdnn_tensor_bcast(src1, dst, src1_contiguous, element_size);
    }

    // zdnn_tensor_pack(src1_packed, src1_contiguous, dst->ne, dst->nb, element_size);

    ggml_zdnn_load_tensor(ztensor_src0, src0_contiguous);
    ggml_zdnn_load_tensor(ztensor_src1, src1_contiguous);

    ZDNN_CHECK(zdnn_op(&ztensor_src0, &ztensor_src1, &ztensor_dst));
    ZDNN_CHECK(zdnn_transform_origtensor(&ztensor_dst, tensor->data));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_src0));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_src1));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_dst));

    if (!ggml_are_same_shape(src0, dst)) {
        ggml_aligned_free(src0_contiguous, dst_buffer_size);
    }

    if (!ggml_are_same_shape(src1, dst)) {
        ggml_aligned_free(src1_contiguous, dst_buffer_size);
    }

    // ggml_aligned_free(src0_packed, dst_buffer_size);
    // ggml_aligned_free(src1_packed, dst_buffer_size);
}

template<zdnn_status (*zdnn_op)(const zdnn_ztensor *, zdnn_ztensor *)>
void ggml_zdnn_op_unary(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor) {
    GGML_UNUSED(ctx);

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * dst  = tensor;

    zdnn_tensor_desc pre_tfm_desc_src0, tfm_desc_src0;
    zdnn_tensor_desc pre_tfm_desc_dst,  tfm_desc_dst;

    zdnn_ztensor ztensor_src0;
    zdnn_ztensor ztensor_dst;

    ggml_zdnn_create_tensor(pre_tfm_desc_src0, tfm_desc_src0, ztensor_src0, src0, dst->ne, ZDNN_NCHW);
    ggml_zdnn_create_tensor(pre_tfm_desc_dst , tfm_desc_dst , ztensor_dst , dst , dst->ne, ZDNN_NCHW);

    ggml_zdnn_load_tensor(ztensor_src0, src0->data);

    ZDNN_CHECK(zdnn_op(&ztensor_src0, &ztensor_dst));
    ZDNN_CHECK(zdnn_transform_origtensor(&ztensor_dst, tensor->data));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_src0));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_dst));
}

static void ggml_zdnn_op_mul_mat(ggml_backend_zdnn_context & ctx,
                                         const ggml_tensor * src0,
                                         const ggml_tensor * src1,
                                               ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const ggml_tensor * weights = src0;
    const ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    zdnn_tensor_desc pre_tfm_desc_weights, tfm_desc_weights;
    zdnn_tensor_desc pre_tfm_desc_inputs,  tfm_desc_inputs;
    zdnn_tensor_desc pre_tfm_desc_bias,    tfm_desc_bias;
    zdnn_tensor_desc pre_tfm_desc_output,  tfm_desc_output;

    zdnn_ztensor ztensor_weights, ztensor_inputs, ztensor_bias, ztensor_output;

    const int64_t weights_rows = ne01;
    const int64_t weights_cols = ne00;
    const int64_t inputs_rows  = ne11;
    const int64_t inputs_cols  = ne10;

    assert(inputs_cols == weights_cols);

    const int64_t output_rows = dst->ne[1];
    const int64_t output_cols = dst->ne[0];

    const int64_t inputs_dim[4]  = { 1, 1, inputs_cols, inputs_rows };
    const int64_t weights_dim[4] = { 1, 1, weights_rows, weights_cols };
    const int64_t bias_dim[4]    = { 1, 1, 1, output_cols };
    const int64_t output_dim[4]  = { 1, 1, output_cols, output_rows };

    ggml_zdnn_create_tensor(pre_tfm_desc_inputs,  tfm_desc_inputs,  ztensor_inputs,  src1, inputs_dim,  ZDNN_2D);
    ggml_zdnn_create_tensor(pre_tfm_desc_weights, tfm_desc_weights, ztensor_weights, src0, weights_dim, ZDNN_2D);
    ggml_zdnn_create_tensor(pre_tfm_desc_bias,    tfm_desc_bias,    ztensor_bias,    dst,  bias_dim,    ZDNN_1D);
    ggml_zdnn_create_tensor(pre_tfm_desc_output,  tfm_desc_output,  ztensor_output,  dst,  output_dim,  ZDNN_2D);

    const size_t weights_size = ggml_element_size(src0);

    void * bias_data = (void *)calloc(output_cols, sizeof(ggml_element_size(dst)));
    void * weights_data_transposed = (void *)ggml_aligned_malloc(weights_cols * weights_rows * weights_size);

    zdnn_transpose((const float *)weights->data,
                   (      float *)weights_data_transposed,
                   weights_cols,
                   weights_rows);

    // for (int i = 0; i < weights_rows; i++) {
    //     for (int j = 0; j < weights_cols; j++) {
    //         memcpy(
    //             (char *)weights_data_transposed + (j * weights_rows + i) * weights_size,
    //             (const char *)weights->data + (i * weights_cols + j) * weights_size,
    //             weights_size
    //         );
    //     }
    // }

    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor_weights, weights_data_transposed));
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor_inputs,  inputs->data));
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor_bias,    bias_data));

    ZDNN_CHECK(zdnn_matmul_op(&ztensor_inputs, &ztensor_weights, &ztensor_bias,
                              MATMUL_OP_ADDITION, &ztensor_output));
    ZDNN_CHECK(zdnn_transform_origtensor(&ztensor_output, output->data));

    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_weights));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_inputs));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_bias));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_output));

    free(bias_data);
    free(weights_data_transposed);
}

inline void ggml_zdnn_mul_mat_dispatch(ggml_backend_zdnn_context & ctx,
                                               const ggml_tensor * src0,
                                               const ggml_tensor * src1,
                                                     ggml_tensor * dst) {
    GGML_UNUSED(ctx);

    bool use_mul_mat_vec =
        (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % 2 == 0 && src1->ne[1] == 1;
    bool use_mul_mat_vec_q =
        ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_q =
        ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // debug helpers
    // GGML_LOG_INFO("%s: use_mul_mat_vec   = %d\n", __func__, use_mul_mat_vec);
    // GGML_LOG_INFO("%s: use_mul_mat_vec_q = %d\n", __func__, use_mul_mat_vec_q);
    // GGML_LOG_INFO("%s: use_mul_mat_q     = %d\n", __func__, use_mul_mat_q);
    // GGML_LOG_INFO("%s: src0: %8d %8d %8d %8d\n", __func__, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    // GGML_LOG_INFO("%s:       %8d %8d %8d %8d\n", __func__, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    // GGML_LOG_INFO("%s: src1: %8d %8d %8d %8d\n", __func__, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    // GGML_LOG_INFO("%s:       %8d %8d %8d %8d\n", __func__, src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    // GGML_LOG_INFO("%s: src0 is contiguous %d, transposed %d, type = %s, name = %s\n", __func__, ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    // GGML_LOG_INFO("%s: src1 is contiguous %d, transposed %d, type = %s, name = %s\n", __func__, ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1)
        && src1->ne[2] * src1->ne[3] > 1) {
        // general KQ + KQV multi-batch
        GGML_LOG_INFO("%s: using zdnn_mul_mat_batched for KQ + KQV multi-batch\n", __func__);
        // ggml_zdnn_mul_mat_batched(ctx, src0, src1, dst);
    } else if (use_mul_mat_vec) {
        GGML_LOG_INFO("%s: using zdnn_op_mul_mat_vec for vector multiplication\n", __func__);
        // ggml_zdnn_op_mul_mat(ctx, src0, src1, dst, ggml_zdnn_op_mul_mat_vec, nullptr);
    } else if (use_mul_mat_vec_q) {
        GGML_LOG_INFO("%s: using zdnn_op_mul_mat_vec_q for quantized vector multiplication\n", __func__);
        // ggml_zdnn_op_mul_mat(ctx, src0, src1, dst, ggml_zdnn_op_mul_mat_vec_q, ggml_zdnn_quantize_row_q8_1);
    } else if (use_mul_mat_q) {
        GGML_LOG_INFO("%s: using zdnn_op_mul_mat_q for quantized matrix multiplication\n", __func__);
        // ggml_zdnn_op_mul_mat(ctx, src0, src1, dst, ggml_zdnn_op_mul_mat_q, ggml_zdnn_quantize_mmq_q8_1);
    } else {
        // GGML_LOG_INFO("%s: using zdnn_op_mul_mat for general matrix multiplication\n", __func__);
        ggml_zdnn_op_mul_mat(ctx, src0, src1, dst);
    }
}

inline bool ggml_zdnn_compute_forward(ggml_backend_zdnn_context & ctx,
                                                    ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ADD:
            //! NOTE: Tested OK
            ggml_zdnn_op_bin<zdnn_add>(ctx, dst);
            break;
        case GGML_OP_ADD1:
            return false;
        case GGML_OP_SUB:
            //! NOTE: Tested did not hit this
            ggml_zdnn_op_bin<zdnn_sub>(ctx, dst);
            break;
        case GGML_OP_MUL:
            //! NOTE: Tested OK
            ggml_zdnn_op_bin<zdnn_mul>(ctx, dst);
            break;
        case GGML_OP_DIV:
            //! NOTE: Tested did not hit this
            ggml_zdnn_op_bin<zdnn_div>(ctx, dst);
            break;
        case GGML_OP_SQRT:
            //! NOTE: Tested did not hit this
            ggml_zdnn_op_unary<zdnn_sqrt>(ctx, dst);
            break;
        case GGML_OP_LOG:
            //! NOTE: Tested did not hit this
            ggml_zdnn_op_unary<zdnn_log>(ctx, dst);
            break;
        case GGML_OP_NORM:
            //! NOTE: Tested did not hit this
            ggml_zdnn_op_bin<zdnn_norm>(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_zdnn_mul_mat_dispatch(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_SOFT_MAX:
            // ggml_zdnn_op_activation<zdnn_softmax>(ctx, dst);
            return false;
        case GGML_OP_LEAKY_RELU:
            // ggml_zdnn_op_activation<zdnn_leaky_relu>(ctx, dst);
            return false;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                    return false;
                case GGML_UNARY_OP_TANH:
                    //! NOTE: Tested did not hit this
                    ggml_zdnn_op_unary<zdnn_tanh>(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    return false;
                case GGML_UNARY_OP_RELU:
                    // ggml_zdnn_op_activation<zdnn_relu>(ctx, dst);
                    return false;
                case GGML_UNARY_OP_SIGMOID:
                    //! NOTE: Tested did not hit this
                    ggml_zdnn_op_unary<zdnn_sigmoid>(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    //! NOTE: Tested did not hit this
                    ggml_zdnn_op_unary<zdnn_gelu>(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                    return false;
                case GGML_UNARY_OP_EXP:
                    //! NOTE: Tested did not hit this
                    ggml_zdnn_op_unary<zdnn_exp>(ctx, dst);
                    break;
                default:
                    return false;
            }
        default:
            return false;
    }

    return true;
}

// --------------------------------------------------------------------------
// Backend buffer type
// --------------------------------------------------------------------------


// --------------------------------------------------------------------------
// Backend buffer
// --------------------------------------------------------------------------


// --------------------------------------------------------------------------
// Backend (stream)
// --------------------------------------------------------------------------
static const char * ggml_backend_zdnn_get_name(ggml_backend_t backend) {
    return GGML_ZDNN_BACKEND_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_zdnn_free(ggml_backend_t backend) {
    GGML_ASSERT(backend != nullptr);

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (backend == nullptr || cgraph == nullptr) {
        return GGML_STATUS_FAILED;
    }

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node)
            || node->op == GGML_OP_NONE
            || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE
            || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        bool ok = ggml_zdnn_compute_forward(*ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, node->name, ggml_op_name(node->op));
        }

        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name            = */ ggml_backend_zdnn_get_name,
    /* .free                = */ ggml_backend_zdnn_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_zdnn_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
};

static ggml_guid_t ggml_backend_zdnn_guid(void) {
    // guid spells out IBM-NNPA-ACCELER
    static ggml_guid guid = { 0x49, 0x42, 0x4D, 0x2D, 0x4E, 0x4E, 0x50, 0x41, 0x2D, 0x41, 0x43, 0x43, 0x45, 0x4C, 0x45, 0x52 };

    return &guid;
}

ggml_backend_t ggml_backend_zdnn_init(void) {
#ifdef STATIC_LIB
    zdnn_init();
#endif

    ggml_backend_zdnn_context * ctx = new ggml_backend_zdnn_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid        = */ ggml_backend_zdnn_guid(),
        /* .interface   = */ ggml_backend_zdnn_i,
        /* .device      = */ ggml_backend_reg_dev_get(ggml_backend_zdnn_reg(), 0),
        /* .context     = */ ctx,
    };

    return backend;
}

// --------------------------------------------------------------------------
// Backend device
// --------------------------------------------------------------------------
static const char * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    return GGML_ZDNN_BACKEND_NAME;

    GGML_UNUSED(dev);
}

static const char * ggml_backend_zdnn_device_get_desc(ggml_backend_dev_t dev) {
    return GGML_ZDNN_BACKEND_NAME;

    GGML_UNUSED(dev);
}

static void ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO: determine if we should report system memory
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name         = ggml_backend_zdnn_device_get_name(dev);
    props->description  = ggml_backend_zdnn_device_get_desc(dev);
    props->type         = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                   = */ false,
        /* .host_buffer             = */ false,
        /* .buffer_from_host_ptr    = */ true,
        /* .events                  = */ false,
    };
}

static ggml_backend_t ggml_backend_zdnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_zdnn_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type(); // TODO: verify if we should use CPU buffer

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_zdnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        // GGML required ops
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;

        // zDNN ops
        case GGML_OP_ADD:
            // zDNN only supports same-shape for element-wise ops
            // TODO: support manual broadcasting
            // if (!ggml_are_same_shape(src0, src1)) return false;
            break;
        case GGML_OP_ADD1:
            return false;
        case GGML_OP_SUB:
            // zDNN only supports same-shape for element-wise ops
            // TODO: support manual broadcasting
            // if (!ggml_are_same_shape(src0, src1)) return false;
            break;
        case GGML_OP_MUL:
            // zDNN only supports same-shape for element-wise ops
            // TODO: support manual broadcasting
            // if (!ggml_are_same_shape(src0, src1)) return false;
            break;
        case GGML_OP_DIV:
            // zDNN only supports same-shape for element-wise ops
            // TODO: support manual broadcasting
            // if (!ggml_are_same_shape(src0, src1)) return false;
            break;
        case GGML_OP_SQRT:
            break;
        case GGML_OP_LOG:
            return false;
        case GGML_OP_NORM:
            break;
        case GGML_OP_MUL_MAT:
            {
                const struct ggml_tensor * a = op->src[0];
                const struct ggml_tensor * b = op->src[1];

                // Note: zDNN cannot handle strided tensors as of now
                // See: https://github.com/IBM/zDNN/issues/37
                if ((b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) ||
                    !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
                    a->ne[0] > 32768 || a->ne[1] > 32768 ||
                    b->ne[0] > 32768 || b->ne[1] > 32768) {
                    return false;
                }

                if ((a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16)
                    && b->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32
                    && op->ne[0] % 2 == 0 && op->ne[1] == 1) {
                    return false;
                }

                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_I8:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_LEAKY_RELU:
            return false; // TODO: disable all support first to showcase device reg
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                    return false;
                case GGML_UNARY_OP_TANH:
                    break;
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                    return false;
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                    return false;
                case GGML_UNARY_OP_EXP:
                    break;
                default:
                    return false;
            }
        default:
            return false;
    }

    return true;

    GGML_UNUSED(dev);
}

static bool ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name                = */ ggml_backend_zdnn_device_get_name,
    /* .get_description         = */ ggml_backend_zdnn_device_get_desc,
    /* .get_memory              = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type                = */ ggml_backend_zdnn_device_get_type,
    /* .get_props               = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend            = */ ggml_backend_zdnn_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type    = */ NULL,
    /* .buffer_from_host_ptr    = */ ggml_backend_zdnn_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft           = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op              = */ NULL, // TODO: decide if we should impl
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

// --------------------------------------------------------------------------
// Backend (reg)
// --------------------------------------------------------------------------
static const char * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_ZDNN_BACKEND_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_zdnn_reg_get_device_count(ggml_backend_reg_t reg) {
    // TODO: implement support for multiple zAIUs
    // Theoretically, customers can create an LPAR that spans across
    // multiple drawers and utilise all zAIU accelerators within those
    // drawers. But we want to ensure that zAIU is working for at least
    // 1 processor before we implement support for additionals.
    if (zdnn_is_nnpa_installed()) {
        return 1;
    }

    return 0;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_zdnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_zdnn_device = {
        /* .interface   = */ ggml_backend_zdnn_device_i,
        /* .register    = */ reg,
        /* .context     = */ nullptr,
    };

    return &ggml_backend_zdnn_device;
}

static void * ggml_backend_zdnn_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return nullptr;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name            = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count    = */ ggml_backend_zdnn_reg_get_device_count,
    /* .get_device          = */ ggml_backend_zdnn_reg_get_device,
    /* .get_proc_address    = */ ggml_backend_zdnn_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zdnn_reg(void) {
    static struct ggml_backend_reg ggml_backend_zdnn_reg = {
        /* .api_version = */ GGML_ZDNN_BACKEND_VERSION,
        /* .interface   = */ ggml_backend_zdnn_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_zdnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
