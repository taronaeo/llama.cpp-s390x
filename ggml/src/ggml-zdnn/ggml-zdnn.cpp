#include "ggml-impl.h"
#include "ggml-zdnn.h"
#include "ggml-backend-impl.h"

#include "ggml-zdnn/zdnn.h"
#include "ggml-zdnn/ggml-zdnn-impl.h"

#include <string>
#include <memory>
#include <stdint.h>
#include <csignal>

// --------------------------------------------------------------------------
// zDNN Internal Helper Functions
// --------------------------------------------------------------------------
void zdnn_tensor_pack(const struct ggml_tensor * src,
                                          void * src_data,
                                          void * dst_data,
                                        size_t   element_size) {
    const int64_t src_w = src->ne[0];
    const int64_t src_h = src->ne[1];
    const int64_t src_c = src->ne[2];
    const int64_t src_n = src->ne[3];

    const int64_t total_elements = src_w * src_h * src_c * src_n;
    const size_t packed_size_bytes = total_elements * element_size;

    const char * src_ptr = (const char *)src_data;
          char * dst_ptr = (      char *)dst_data;

    for (int64_t n = 0; n < src_n; ++n) {
        for (int64_t c = 0; c < src_c; ++c) {
            for (int64_t h = 0; h < src_h; ++h) {
                for (int64_t w = 0; w < src_w; ++w) {
                    const size_t src_offset = w * src->nb[0]
                                            + h * src->nb[1]
                                            + c * src->nb[2]
                                            + n * src->nb[3];

                    const char * src_element = (const char *)(src_ptr + src_offset);
                    memcpy(dst_ptr, src_element, element_size);
                    dst_ptr += element_size;
                }
            }
        }
    }
}

void zdnn_tensor_bcast(const struct ggml_tensor * src,
                       const struct ggml_tensor * dst,
                                           void * dst_data,
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
          char * dst_ptr = (      char *)dst_data;

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


// --------------------------------------------------------------------------
// zDNN Interfacing API
// --------------------------------------------------------------------------
static zdnn_data_types ggml_zdnn_type_mapping(ggml_type type) {
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

void ggml_zdnn_create_tensor(const ggml_tensor      * src,
                             const int64_t          * ne,
                                   zdnn_tensor_desc & pre_tfm_desc,
                                   zdnn_tensor_desc & tfm_desc,
                                   zdnn_ztensor     & ztensor) {

    zdnn_init_pre_transformed_desc(
        ZDNN_NCHW,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

void ggml_zdnn_load_tensor(const ggml_tensor  * tensor,
                                 zdnn_ztensor & ztensor) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, tensor->data));
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

    ggml_zdnn_create_tensor(src0, dst->ne, pre_tfm_desc_src0, tfm_desc_src0, ztensor_src0);
    ggml_zdnn_create_tensor(src1, dst->ne, pre_tfm_desc_src1, tfm_desc_src1, ztensor_src1);
    ggml_zdnn_create_tensor(dst , dst->ne, pre_tfm_desc_dst , tfm_desc_dst , ztensor_dst );

    void * src0_contiguous = nullptr;
    void * src1_contiguous = nullptr;

    size_t element_size = ggml_element_size(dst);
    size_t dst_buffer_size = ggml_nelements(dst) * element_size;

    void * src0_packed = ggml_aligned_malloc(dst_buffer_size);
    void * src1_packed = ggml_aligned_malloc(dst_buffer_size);

    if (ggml_are_same_shape(src0, dst)) {
        src0_contiguous = (void *)src0->data;
    } else {
        src0_contiguous = ggml_aligned_malloc(dst_buffer_size);
        zdnn_tensor_bcast(src0, dst, src0_contiguous, element_size);
    }

    zdnn_tensor_pack(dst, src0_contiguous, src0_packed, element_size);

    if (ggml_are_same_shape(src1, dst)) {
        src1_contiguous = (void *)src1->data;
    } else {
        src1_contiguous = ggml_aligned_malloc(dst_buffer_size);
        zdnn_tensor_bcast(src1, dst, src1_contiguous, element_size);
    }

    zdnn_tensor_pack(dst, src1_contiguous, src1_packed, element_size);

    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor_src0, src0_packed));
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor_src1, src1_packed));

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

    ggml_aligned_free(src0_packed, dst_buffer_size);
    ggml_aligned_free(src1_packed, dst_buffer_size);
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

    ggml_zdnn_create_tensor(src0, dst->ne, pre_tfm_desc_src0, tfm_desc_src0, ztensor_src0);
    ggml_zdnn_create_tensor(dst , dst->ne, pre_tfm_desc_dst , tfm_desc_dst , ztensor_dst );

    ggml_zdnn_load_tensor(src0, ztensor_src0);

    ZDNN_CHECK(zdnn_op(&ztensor_src0, &ztensor_dst));
    ZDNN_CHECK(zdnn_transform_origtensor(&ztensor_dst, tensor->data));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_src0));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_dst));
}

static bool ggml_zdnn_compute_forward(struct ggml_backend_zdnn_context & ctx,
                                      struct               ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
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
    return 1;

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
