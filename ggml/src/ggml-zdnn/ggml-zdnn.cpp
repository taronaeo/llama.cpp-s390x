#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-zdnn/zdnn.h"
#include "ggml-zdnn/ggml-zdnn-impl.h"

// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Backend Buffer
// --------------------------------------------------------------------------

static void ggml_backend_zdnn_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, buffer->size);
}

static void * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // TODO: Work on this
}

static void ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_zdnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_zdnn_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ nullptr,
};

static const ggml_backend_buffer_i ggml_backend_zdnn_buffer_from_ptr_i = {
    /* .free_buffer   = */ nullptr, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_zdnn_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ nullptr,
};

// --------------------------------------------------------------------------
// Backend Buffer Type
// --------------------------------------------------------------------------

static const char * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ZDNN_BACKEND_NAME;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, data, size);
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

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            case GGML_OP_MUL_MAT:
                GGML_LOG_INFO("%s: processing node %d: %s\n", __func__, i, ggml_op_desc(node));
                break;

            case GGML_OP_OUT_PROD:
                GGML_LOG_INFO("%s: processing node %d: %s\n", __func__, i, ggml_op_desc(node));
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
    GGML_UNUSED(ctx);
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
        case GGML_OP_OUT_PROD:
            return true;

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
    return 1;  // zDNN backend currently supports only one device

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
