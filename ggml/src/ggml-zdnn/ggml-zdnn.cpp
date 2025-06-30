#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-zdnn/zdnn.h"
#include "ggml-zdnn/ggml-zdnn-impl.h"

#include <csignal>

struct ggml_backend_zdnn_buffer_context {
    zdnn_tensor_desc pre_transform_desc;
    zdnn_tensor_desc transform_desc;
    zdnn_ztensor ztensor;

    struct ggml_backend_zdnn_buffer_context * src[GGML_MAX_SRC];  // for src tensors that went through CPU instead of zDNN
    struct ggml_backend_zdnn_buffer_context * extra;  // for bias, etc.
};

// --------------------------------------------------------------------------
// Utilities
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
    }
}


// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------

inline void ggml_zdnn_op_mul_mat(ggml_backend_zdnn_context & ctx,
                                         const ggml_tensor * src0,
                                         const ggml_tensor * src1,
                                               ggml_tensor * dst) {
    const ggml_backend_zdnn_buffer_context * weights_ctx = (ggml_backend_zdnn_buffer_context *)src0->buffer->context;
    const ggml_backend_zdnn_buffer_context * inputs_ctx  = (ggml_backend_zdnn_buffer_context *)src1->buffer->context;
          ggml_backend_zdnn_buffer_context * output_ctx  = (ggml_backend_zdnn_buffer_context *)dst->buffer->context;

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

    const int64_t weights_rows = ne01;
    const int64_t weights_cols = ne00;
    const int64_t inputs_rows  = ne11;
    const int64_t inputs_cols  = ne10;

    assert(inputs_cols == weights_cols);

    const int64_t output_rows = ne1;
    const int64_t output_cols = ne0;

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_ctx->ztensor,
                                        &weights_ctx->ztensor,
                                        &output_ctx->extra->ztensor,
                                        false, true, MATMUL_OP_ADDITION,
                                        &output_ctx->ztensor));
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
        case GGML_OP_MUL_MAT:
            ggml_zdnn_mul_mat_dispatch(ctx, dst->src[0], dst->src[1], dst);
            break;

        default:
            return false;
    }

    return true;
}

// --------------------------------------------------------------------------
// Backend Buffer
// --------------------------------------------------------------------------

static void ggml_backend_zdnn_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    if (ctx->ztensor.pre_transformed_desc != nullptr) ZDNN_CHECK(zdnn_free_ztensor_buffer(&ctx->ztensor));

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (ctx->src[i] != nullptr) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&ctx->src[i]->ztensor));
            delete ctx->src[i];
        }
    }

    delete ctx;
}

static void * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            {
                // needed because some buffers are CPU and zDNN requires all tensors to be transformed
                for (int i = 0; i < GGML_MAX_SRC; i++) {
                    if (tensor->src[i] != nullptr
                        && tensor->src[i]->buffer->buft == ggml_backend_cpu_buffer_type()) {
                        ggml_backend_zdnn_buffer_context * src_ctx = new ggml_backend_zdnn_buffer_context{};
                        if (!src_ctx) {
                            GGML_ABORT("%s: fatal: memory allocation for src_ctx failed", __func__);
                        }
                        zdnn_init_pre_transformed_desc(
                            ZDNN_2D,
                            ggml_zdnn_type_mapping(tensor->src[i]->type),
                            &src_ctx->pre_transform_desc,
                            1, 1, tensor->src[i]->ne[1], tensor->src[i]->ne[0]
                        );

                        ZDNN_CHECK(zdnn_generate_transformed_desc(&src_ctx->pre_transform_desc, &src_ctx->transform_desc));
                        if (tensor->src[i]->buffer->context != nullptr) {
                            delete (ggml_backend_zdnn_buffer_context *)tensor->src[i]->buffer->context;
                        }
                        tensor->src[i]->buffer->context = src_ctx;

                        ctx->src[i] = src_ctx;
                    }
                }

                if (tensor->extra != nullptr) {
                    ggml_backend_zdnn_buffer_context * bias_ctx = (ggml_backend_zdnn_buffer_context *)tensor->extra;
                    zdnn_init_pre_transformed_desc(
                        ZDNN_1D,
                        ggml_zdnn_type_mapping(tensor->type),
                        &bias_ctx->pre_transform_desc,
                        1, 1, 1, tensor->ne[0]
                    );
                    ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_ctx->pre_transform_desc, &bias_ctx->transform_desc));
                    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias_ctx->pre_transform_desc, &bias_ctx->transform_desc, &bias_ctx->ztensor));

                    if (tensor->extra != nullptr) {
                        ggml_backend_zdnn_buffer_context * old_ctx = (ggml_backend_zdnn_buffer_context *)tensor->extra;
                        ZDNN_CHECK(zdnn_free_ztensor_buffer(&old_ctx->ztensor));
                        delete old_ctx;
                    }

                    tensor->extra = bias_ctx;
                    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias_ctx->pre_transform_desc, &bias_ctx->transform_desc, &bias_ctx->ztensor));
                }

                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &ctx->pre_transform_desc,
                    1, 1, tensor->ne[1], tensor->ne[0]
                );
            } break;
        default:
            zdnn_init_pre_transformed_desc(
                ZDNN_NCHW,
                ggml_zdnn_type_mapping(tensor->type),
                &ctx->pre_transform_desc,
                tensor->ne[3], tensor->ne[2],
                tensor->ne[1], tensor->ne[0]
            );
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&ctx->pre_transform_desc, &ctx->transform_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&ctx->pre_transform_desc, &ctx->transform_desc, &ctx->ztensor));

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    // memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // memcpy((char *)tensor->data + offset, data, size);
    zdnn_status status;
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    if (ctx->ztensor.is_transformed) {
        zdnn_reset_ztensor(&ctx->ztensor);
        // return;  // TODO: Check if we should reset the ztensor or return
    }

    status = zdnn_transform_ztensor(&ctx->ztensor, (char *)data + offset);
    if (status == ZDNN_OK) {
        return;
    } else if (status == ZDNN_INVALID_FORMAT) {
        GGML_LOG_INFO("%s: ZDNN_INVALID_FORMAT\n", __func__);
        return;
    } else if (status == ZDNN_INVALID_LAYOUT) {
        GGML_LOG_INFO("%s: ZDNN_INVALID_LAYOUT\n", __func__);
        return;
    } else if (status == ZDNN_INVALID_TYPE) {
        GGML_LOG_INFO("%s: ZDNN_INVALID_TYPE\n", __func__);
        return;
    } else if (status == ZDNN_INVALID_BUFFER) {
        GGML_LOG_INFO("%s: ZDNN_INVALID_BUFFER\n", __func__);
        return;
    } else if (status == ZDNN_INVALID_SHAPE) {
        GGML_LOG_INFO("%s: ZDNN_INVALID_SHAPE\n", __func__);
        return;
    } else if (status == ZDNN_INVALID_STATE) {
        GGML_LOG_INFO("%s: ZDNN_INVALID_STATE\n", __func__);
        return;
    } else if (status == ZDNN_CONVERT_FAILURE) {
        GGML_LOG_INFO("%s: ZDNN_CONVERT_FAILURE\n", __func__);
        return;
    } else if (status == ZDNN_FUNC_RC_F000) {
        GGML_LOG_INFO("%s: ZDNN_FUNC_RC_F000, unsupported transformation function\n", __func__);
        return;
    }
}

static void ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // memcpy(data, (const char *)tensor->data + offset, size);
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    ZDNN_CHECK(zdnn_transform_origtensor(&ctx->ztensor, (char *)data + offset));
}

static bool ggml_backend_zdnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // if (ggml_backend_buffer_is_host(src->buffer)) {
    //     memcpy(dst->data, src->data, ggml_nbytes(src));
    //     return true;
    // }
    // return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // memset(buffer->context, value, buffer->size);
}

static const ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ nullptr,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ nullptr,
    /* .reset         = */ nullptr,
};

static const ggml_backend_buffer_i ggml_backend_zdnn_buffer_from_ptr_i = {
    /* .free_buffer   = */ nullptr, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ nullptr,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ nullptr,
    /* .reset         = */ nullptr,
};

// --------------------------------------------------------------------------
// Backend Buffer Type
// --------------------------------------------------------------------------

struct ggml_backend_zdnn_buffer_type_context {};

static const char * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ZDNN_BACKEND_NAME;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_type_context * ctx = (ggml_backend_zdnn_buffer_type_context *)buft->context;

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft,
        ggml_backend_zdnn_buffer_i,
        new ggml_backend_zdnn_buffer_context{},
        size
    );

    return buffer;
}

static size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(void) {
    static ggml_backend_buffer_type ggml_backend_zdnn_buffer_type = {
        /* .iface   = */ {
            /* .get_name        = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer    = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment   = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size    = */ nullptr,
            /* .get_alloc_size  = */ nullptr,
            /* .is_host         = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    return &ggml_backend_zdnn_buffer_type;
}

static const char * ggml_backend_zdnn_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ZDNN_BACKEND_NAME "_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_zdnn_buffer_type = {
        /* .iface   = */ {
            /* .get_name        = */ ggml_backend_zdnn_buffer_from_ptr_type_get_name,
            /* .alloc_buffer    = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment   = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size    = */ nullptr,
            /* .get_alloc_size  = */ nullptr,
            /* .is_host         = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    return &ggml_backend_zdnn_buffer_type;
}

ggml_backend_buffer_t ggml_backend_zdnn_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_zdnn_buffer_from_ptr_type(), ggml_backend_zdnn_buffer_from_ptr_i, ptr, size);
}


// --------------------------------------------------------------------------
// Backend Interface
// --------------------------------------------------------------------------

static const char * ggml_backend_zdnn_get_name(ggml_backend_t backend) {
    return GGML_ZDNN_BACKEND_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
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
                           __func__, ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }

        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_get_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ nullptr,
    /* .get_tensor_async       = */ nullptr,
    /* .cpy_tensor_async       = */ nullptr,
    /* .synchronize            = */ nullptr,
    /* .graph_plan_create      = */ nullptr,
    /* .graph_plan_free        = */ nullptr,
    /* .graph_plan_update      = */ nullptr,
    /* .graph_plan_compute     = */ nullptr,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ nullptr,
    /* .event_wait             = */ nullptr,
};

static ggml_guid_t ggml_backend_zdnn_guid(void) {
    // guid spells out IBM-NNPA-ACCELER
    static ggml_guid guid = { 0x49, 0x42, 0x4D, 0x2D, 0x4E, 0x4E, 0x50, 0x41,
                              0x2D, 0x41, 0x43, 0x43, 0x45, 0x4C, 0x45, 0x52 };
    return &guid;
}

bool ggml_backend_is_zdnn(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

// --------------------------------------------------------------------------
// Backend Device Interface
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
    return ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_zdnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_zdnn_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
            {
                const ggml_tensor * a = op->src[0];
                const ggml_tensor * b = op->src[1];

                if ((b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) ||
                    !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
                    a->ne[0] > 32768 || a->ne[1] > 32768 ||
                    b->ne[0] > 32768 || b->ne[1] > 32768) {
                    return false;
                }

                // Disable batched matrix multiplication for now
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
        case GGML_OP_OUT_PROD:
            return false;

        default:
            return false;
    }

    GGML_UNUSED(dev);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
}

static bool ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name                = */ ggml_backend_zdnn_device_get_name,
    /* .get_description         = */ ggml_backend_zdnn_device_get_desc,
    /* .get_memory              = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type                = */ ggml_backend_zdnn_device_get_type,
    /* .get_props               = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend            = */ ggml_backend_zdnn_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type    = */ nullptr, // TODO: decide if we should impl
    /* .buffer_from_host_ptr    = */ ggml_backend_zdnn_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft           = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op              = */ nullptr, // TODO: decide if we should impl
    /* .event_new               = */ nullptr,
    /* .event_free              = */ nullptr,
    /* .event_synchronize       = */ nullptr,
};

// --------------------------------------------------------------------------
// Backend Registration Interface
// --------------------------------------------------------------------------

static const char * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_ZDNN_BACKEND_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_zdnn_reg_get_device_count(ggml_backend_reg_t reg) {
    if (zdnn_is_nnpa_installed()) {
        return 1;  // zDNN backend currently supports only one device
    }

    return 0;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_zdnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_zdnn_device = {
        /* .iface   = */ ggml_backend_zdnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return nullptr;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_get_device_count,
    /* .get_device       = */ ggml_backend_zdnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address,
};

// --------------------------------------------------------------------------
// Backend Registry Interface
// --------------------------------------------------------------------------

ggml_backend_reg_t ggml_backend_zdnn_reg(void) {
    static ggml_backend_reg ggml_backend_zdnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_zdnn_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_zdnn_reg;
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

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
