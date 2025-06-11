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
/**
 * @brief Packs a multi-dimensional tensor from a source buffer into a contiguous destination buffer.
 *
 * This function iterates over all elements of a 4D tensor (with dimensions width, height, channels, and batch)
 * and copies each element from the source buffer to the destination buffer in a packed, contiguous format.
 * The source buffer is accessed using the provided strides (`nb`) and element size.
 *
 * @param dst_buffer    Pointer to the destination buffer where the packed tensor will be stored.
 * @param src_buffer    Pointer to the source buffer containing the original tensor data.
 * @param ne            Array of 4 integers specifying the size of each tensor dimension (width, height, channels, batch).
 * @param nb            Array of 4 sizes specifying the stride (in bytes) for each dimension in the source buffer.
 * @param element_size  Size (in bytes) of each tensor element.
 *
 * @note This function is currently unused due to poor performance.
 */
void zdnn_tensor_pack(         void * dst_buffer,
                      const    void * src_buffer,
                      const int64_t * ne,
                      const  size_t * nb,
                             size_t   element_size);

/**
 * @brief Broadcasts the contents of a source tensor to a destination tensor buffer, supporting broadcasting semantics.
 *
 * This function copies data from the source tensor (`src`) to the destination tensor (`dst_data`),
 * following broadcasting rules for each dimension. If a dimension in the source tensor is 1, its value
 * is broadcast (repeated) along that dimension in the destination tensor. The function assumes both
 * tensors are 4-dimensional and uses the provided element size for memory copying.
 *
 * @param src           Pointer to the source tensor structure.
 * @param dst           Pointer to the destination tensor structure (used for shape/stride info only).
 * @param dst_buffer    Pointer to the destination data buffer where the broadcasted tensor will be written.
 * @param element_size  Size in bytes of each tensor element.
 */
void zdnn_tensor_bcast(const struct ggml_tensor * src,
                       const struct ggml_tensor * dst,
                                           void * dst_data,
                                         size_t   element_size);

// --------------------------------------------------------------------------
// zDNN Interfacing API
// --------------------------------------------------------------------------
/**
 * @brief Maps a GGML tensor type to the corresponding zDNN data type.
 *
 * This function takes a GGML tensor type (ggml_type) and returns the
 * equivalent zdnn_data_types enumeration value. If the provided type
 * does not have a corresponding zDNN data type, the function aborts
 * execution with an error message.
 *
 * @param type The GGML tensor type to map.
 * @return The corresponding zdnn_data_types value.
 *
 * @note Supported mappings:
 *
 *   - GGML_TYPE_F32  -> FP32
 *
 *   - GGML_TYPE_F16  -> FP16
 *
 *   - GGML_TYPE_BF16 -> BFLOAT
 *
 *   - GGML_TYPE_I8   -> INT8
 *
 *   - GGML_TYPE_I32  -> INT32
 *
 *   - GGML_TYPE_Q8_0 -> INT8
 *
 * @throws Aborts the program if the type is not supported.
 */
static zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

/**
 * @brief Creates and initializes a zDNN tensor from a given GGML tensor.
 *
 * This function sets up the pre-transformed and transformed tensor descriptors
 * and allocates memory for the zDNN tensor using the provided GGML tensor and
 * its shape information.
 *
 * @param src           Pointer to the source GGML tensor.
 * @param ne            Pointer to an array of 4 int64_t values representing the tensor shape
 *                      in the reversed-NCHW format (i.e., WHCN).
 * @param pre_tfm_desc  Reference to a zdnn_tensor_desc to be initialized as the pre-transformed descriptor.
 * @param tfm_desc      Reference to a zdnn_tensor_desc to be initialized as the transformed descriptor.
 * @param ztensor       Reference to a zdnn_ztensor to be initialized and allocated.
 */
void ggml_zdnn_create_tensor(const ggml_tensor      * tensor,
                                   zdnn_tensor_desc & pre_tfm_desc,
                                   zdnn_tensor_desc & tfm_desc,
                                   zdnn_ztensor     & ztensor,
                             const ggml_tensor      * dst);

/**
 * @brief Loads data from a buffer into a zdnn_ztensor.
 *
 * This function takes a pointer to a buffer containing tensor data and loads it into
 * the provided zdnn_ztensor structure by calling zdnn_transform_ztensor. It checks
 * for errors using the ZDNN_CHECK macro.
 *
 * @param buffer   Pointer to the source buffer containing tensor data.
 * @param ztensor  Reference to the zdnn_ztensor structure to be populated.
 */
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
