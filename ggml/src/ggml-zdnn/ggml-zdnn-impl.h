#pragma once

#ifndef GGML_ZDNN_IMPL
#define GGML_ZDNN_IMPL

#include "zdnn.h"
#include "../include/ggml.h"
#include "../include/ggml-zdnn.h"

#include <memory>
#include <vecintrin.h>

#define vec_neg(a)    (-(a))                // Vector Negate
#define vec_add(a, b) ((a) + (b))           // Vector Add
#define vec_sub(a, b) ((a) - (b))           // Vector Subtract
#define vec_mul(a, b) ((a) * (b))           // Vector Multiply
#define vec_div(a, b) ((a) / (b))           // Vector Divide
#define vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#define vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#define vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#define vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#define vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#ifndef vec_and
#define vec_and(a, b) ((a) & (b)) // Vector AND
#endif

#ifndef vec_or
#define vec_or(a, b)  ((a) | (b)) // Vector OR
#endif

#ifndef vec_xor
#define vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#endif

typedef signed char char8x16_t __attribute__((vector_size(16)));
typedef unsigned char uchar8x16_t __attribute__((vector_size(16)));

typedef int8_t  int8x16_t __attribute__((vector_size(16)));
typedef int16_t int16x8_t __attribute__((vector_size(16)));
typedef int32_t int32x4_t __attribute__((vector_size(16)));

typedef uint8_t  uint8x16_t __attribute__((vector_size(16)));
typedef uint16_t uint16x8_t __attribute__((vector_size(16)));
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));

typedef float float32x4_t __attribute__((vector_size(16)));
typedef double double64x2_t __attribute__((vector_size(16)));

typedef signed long long long64x2_t __attribute__((vector_size(16)));
typedef unsigned long long ulong64x2_t __attribute__((vector_size(16)));

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
inline zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

/**
 * @brief Creates and initializes a zDNN tensor from a given GGML tensor.
 *
 * This function sets up the pre-transformed and transformed tensor descriptors
 * and allocates memory for the zDNN tensor using the provided GGML tensor and
 * its shape information.
 *
 * @param pre_tfm_desc  Reference to a zdnn_tensor_desc to be initialized as the pre-transformed descriptor.
 * @param tfm_desc      Reference to a zdnn_tensor_desc to be initialized as the transformed descriptor.
 * @param ztensor       Reference to a zdnn_ztensor to be initialized and allocated.
 * @param src           Pointer to the source GGML tensor.
 * @param ne            Pointer to an array of 4 int64_t values representing the tensor shape
 *                      in the reversed-NCHW format (i.e., WHCN).
 * @param layout        The zdnn_data_layouts enum value representing the desired data layout for the tensor.
 */
inline void ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                                    zdnn_tensor_desc  & tfm_desc,
                                    zdnn_ztensor      & ztensor,
                              const ggml_tensor       * src,
                              const int64_t           * ne,
                              const zdnn_data_layouts   layout);

/**
 * @brief Loads data from a buffer into a zdnn_ztensor.
 *
 * This function takes a pointer to a buffer containing tensor data and loads it into
 * the provided zdnn_ztensor structure by calling zdnn_transform_ztensor. It checks
 * for errors using the ZDNN_CHECK macro.
 *
 * @param ztensor  Reference to the zdnn_ztensor structure to be populated.
 * @param buffer   Pointer to the source buffer containing tensor data.
 */
inline void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor,
                                          void * buffer);

template<zdnn_status (*zdnn_op)(const zdnn_ztensor *, const zdnn_ztensor *, zdnn_ztensor *)>
void ggml_zdnn_op_bin(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);

template<zdnn_status (*zdnn_op)(const zdnn_ztensor *, zdnn_ztensor *)>
void ggml_zdnn_op_unary(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);

static void ggml_zdnn_op_mul_mat(ggml_backend_zdnn_context & ctx,
                                         const ggml_tensor * src0,
                                         const ggml_tensor * src1,
                                               ggml_tensor * dst);

/**
 * @brief Dispatches matrix multiplication operations for zDNN backend based on tensor types and shapes.
 *
 * This function selects the appropriate matrix multiplication routine depending on the types and shapes
 * of the input tensors. It supports various cases including:
 * - Standard matrix multiplication for F16 tensors.
 * - Vector multiplication for F16/BF16 matrices and F32 vectors.
 * - Quantized matrix and vector multiplication for quantized types.
 * - General matrix multiplication as a fallback.
 *
 * @param ctx   zDNN backend context used for operation dispatch.
 * @param src0  Pointer to the first source tensor (matrix).
 * @param src1  Pointer to the second source tensor (matrix or vector).
 * @param dst   Pointer to the destination tensor where the result will be stored.
 */
inline void ggml_zdnn_mul_mat_dispatch(ggml_backend_zdnn_context & ctx,
                                               const ggml_tensor * src0,
                                               const ggml_tensor * src1,
                                                     ggml_tensor * dst);

/**
 * @brief Executes the forward computation for a given tensor operation using the zDNN backend.
 *
 * This function dispatches the computation based on the operation type specified in the destination tensor (`dst->op`).
 * It supports a variety of binary and unary operations, as well as matrix multiplication. For unsupported or unimplemented
 * operations, the function returns false.
 *
 * @param ctx  Reference to the zDNN backend context used for computation.
 * @param dst  Pointer to the destination tensor, which contains the operation type and source tensors.
 * @return true if the operation was successfully dispatched and computed; false if the operation is unsupported or not implemented.
 */
inline bool ggml_zdnn_compute_forward(ggml_backend_zdnn_context & ctx,
                                                    ggml_tensor * dst);

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
