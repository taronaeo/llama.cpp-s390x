#include "ggml-impl.h"
#include "ggml-zdnn.h"
#include "ggml-backend-impl.h"

#include "zdnn.h"

#include <string>
#include <memory>
#include <stdint.h>

struct ggml_backend_zdnn_context {
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
};

// --------------------------------------------------------------------------
// zDNN Interfacing API
// --------------------------------------------------------------------------
zdnn_data_types ggml_zdnn_type_mapping(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return FP32;
        case GGML_TYPE_F16:
            return FP16;
        case GGML_TYPE_I8:
            return INT8;
        case GGML_TYPE_I32:
            return INT32;
        case GGML_TYPE_Q8_0:
            return INT8;
        default:
            GGML_ABORT("unable to determine zTensor data type");
            break;
    }
}

void ggml_zdnn_create_tensor(const ggml_tensor * tensor,
                              zdnn_tensor_desc * pre_tfm_desc,
                              zdnn_tensor_desc * tfm_desc,
                                       int64_t * ne,
                                        size_t * nb,
                                       int64_t   dims,
                                        size_t   offset);

void ggml_zdnn_op_add(ggml_backend_zdnn_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    zdnn_tensor * ztensor_src0;
    zdnn_tensor * ztensor_src1;
    zdnn_tensor * ztensor_dst;
}

static bool ggml_zdnn_compute_forward(ggml_backend_zdnn_context & ctx,
                                      struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ADD:
            ggml_zdnn_op_add(ctx, dst);
            break;
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_NORM:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_RESHAPE: // TODO: verify if should op
        case GGML_OP_SOFT_MAX:
        case GGML_OP_LEAKY_RELU:
            break;
        default:
            GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
    }
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
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

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
    GGML_UNUSED(ptr);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        // GGML required ops
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:

        // zDNN ops
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_NORM:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_RESHAPE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_LEAKY_RELU:
            return false; // TODO: disable all support first to showcase device reg
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_EXP:
                    return ggml_is_contiguous(src0);
                default:
                    return false;
            }
        default:
            return false;
    }

    GGML_UNUSED(dev);
    GGML_UNUSED(src1); // TODO: remove when impl op
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
