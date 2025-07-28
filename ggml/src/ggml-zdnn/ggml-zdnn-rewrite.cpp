#include "zdnn.h"
#include "ggml-zdnn.h"
#include "ggml-zdnn-impl.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <csignal>
#include <unistd.h>

static bool ggml_zdnn_op_mul_mat(struct ggml_backend_zdnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const ggml_tensor * weights = src0;
    const ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_buffer * weights_extra = (ggml_backend_zdnn_buffer *)weights->extra;
    ggml_backend_zdnn_buffer * inputs_extra  = (ggml_backend_zdnn_buffer *)inputs->extra;
    ggml_backend_zdnn_buffer * output_extra  = (ggml_backend_zdnn_buffer *)output->extra;
    ggml_backend_zdnn_buffer * bias_extra    = (ggml_backend_zdnn_buffer *)output_extra->extra;

    const int64_t weights_rows = ne01;
    const int64_t weights_cols = ne00;
    const int64_t inputs_rows  = ne11;
    const int64_t inputs_cols  = ne10;

    assert(inputs_cols == weights_cols);

    const int64_t output_rows = ne1;
    const int64_t output_cols = ne0;

    // have to do this because weights apparently do not go through set_tensor
    if (&weights_extra->ztensor.is_transformed) zdnn_reset_ztensor(&weights_extra->ztensor);
    zdnn_init_pre_transformed_desc(
        ZDNN_2D,
        FP32,
        &weights_extra->pre_tfm_desc,
        weights->ne[1], weights->ne[0]
    );
    ZDNN_CHECK(zdnn_generate_transformed_desc(&weights_extra->pre_tfm_desc, &weights_extra->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&weights_extra->pre_tfm_desc, &weights_extra->tfm_desc, &weights_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_ztensor(&weights_extra->ztensor, weights->data));

    // have to do this here because although it was transformed, the shape is wrong
    if (&inputs_extra->ztensor.is_transformed) zdnn_reset_ztensor(&inputs_extra->ztensor);
    zdnn_init_pre_transformed_desc(
        ZDNN_2D,
        FP32,
        &inputs_extra->pre_tfm_desc,
        inputs->ne[1], inputs->ne[0]
    );
    ZDNN_CHECK(zdnn_generate_transformed_desc(&inputs_extra->pre_tfm_desc, &inputs_extra->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&inputs_extra->pre_tfm_desc, &inputs_extra->tfm_desc, &inputs_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_ztensor(&inputs_extra->ztensor, inputs->data));

    // have to transform the bias ztensor here because only GGML_OP_NONE goes through set_tensor
    ZDNN_CHECK(zdnn_transform_ztensor(&bias_extra->ztensor, bias_extra->data));

    std::raise(SIGINT);
    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        false, true, MATMUL_OP_ADDITION, &output_extra->ztensor));
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->data));
}

static bool ggml_backend_zdnn_compute_forward(struct ggml_backend_zdnn_context * ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            ggml_zdnn_op_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        default:
            return false;
    }

    return true;

    GGML_UNUSED(ctx);
}

//
// globals
//

// initialised in ggml_backend_zdnn_reg
static struct ggml_backend_reg    g_ggml_backend_zdnn_reg;
static struct ggml_backend_device g_ggml_backend_zdnn_device;

// information about an NNPA device
// note: assumes single NNPA device - the default one
static struct ggml_backend_zdnn_device_context {
    int zdnn_device;
    int zdnn_device_ref_count;

    bool has_nnpa_parmblkformat_1;

    int32_t max_dim_idx_size;

    char name[128];
} g_ggml_ctx_dev_main = {
    /* .zdnn_device              = */ 0,
    /* .zdnn_device_ref_count    = */ 0,
    /* .has_nnpa_parmblkformat_1 = */ false,
    /* .max_dim_idx_size         = */ 0,
    /* .name                     = */ "",
};

// acquire
static int ggml_backend_zdnn_device_acq(struct ggml_backend_zdnn_device_context * ctx) {
    assert(ctx != NULL);

    if (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    if (ctx->zdnn_device) {
        // ctx->has_nnpa_parmblkformat_1 = zdnn_has_nnpa_parmblkformat_1(ctx->zdnn_device);
        ctx->max_dim_idx_size = zdnn_get_nnpa_max_dim_idx_size();

        strncpy(ctx->name, GGML_ZDNN_NAME, sizeof(ctx->name) - 1);
        ctx->name[sizeof(ctx->name) - 1] = '\0';
    }

    ctx->zdnn_device_ref_count++;
    return ctx->zdnn_device;
}

// release
static void ggml_backend_zdnn_device_rel(struct ggml_backend_zdnn_device_context * ctx) {
    assert(ctx != NULL);
    assert(ctx->zdnn_device_ref_count > 0);

    ctx->zdnn_device_ref_count--;
}

struct ggml_backend_zdnn_context {
    int device;

    struct ggml_cgraph * gf;
};

static struct ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    struct ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)calloc(1, sizeof(struct ggml_backend_zdnn_context));
    struct ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    int device = ctx_dev->zdnn_device;

    GGML_LOG_INFO("%s: picking default device: %d\n", __func__, device);

    ctx->device = device;

    // GGML_LOG_INFO("%s: NNPA Name: %s\n", __func__, )
    // GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_nnpa_parmblkformat_1 ? "true" : "false");

    return ctx;
}

static void ggml_zdnn_free(struct ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);
    free(ctx);
}

struct ggml_backend_zdnn_buffer_context {
    void * all_data;
    size_t all_size;
    bool owned;

    int n_buffers;
    struct ggml_backend_zdnn_buffer buffers[999999];  // TODO: CHANGE TO VECTOR
};

// finds the zTensor that contains the tensor data
// the assumption is that there is a 1-to-1 mapping between the host and NNPA
// device buffers, so we can find the zTensor buffer based on the host memory pointer
static zdnn_ztensor * ggml_zdnn_get_buffer(struct ggml_tensor * t, size_t * offset) {
    const int64_t tsize = ggml_nbytes(t);

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    struct ggml_backend_zdnn_buffer_context * buf_ctx = (struct ggml_backend_zdnn_buffer_context *)buffer->context;

    // find the view that contains the tensor fully
    for (int i = 0; i < buf_ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t)t->data - (int64_t)buf_ctx->buffers[i].data;

        if (ioffs >= 0 && ioffs + tsize <= (int64_t)buf_ctx->buffers[i].size) {
            *offset = (size_t)ioffs;

            return &buf_ctx->buffers[i].ztensor;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return NULL;
}

static bool ggml_zdnn_supports_op(const struct ggml_backend_zdnn_device_context * ctx_dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONCAT:
            return true;

        case GGML_OP_MUL_MAT:
            {
                GGML_TENSOR_BINARY_OP_LOCALS

                const int32_t max_dim_idx_size = ctx_dev->max_dim_idx_size;

                return ggml_is_contiguous(src0) &&
                       ggml_is_contiguous(src1) &&
                       src1->type == GGML_TYPE_F32 &&
                       (ne0 <= max_dim_idx_size && ne1 <= max_dim_idx_size && ne10 <= max_dim_idx_size) &&
                       (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
            } break;

        default:
            return false;
    }

    GGML_UNUSED(ctx_dev);
}

static enum ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * gf) {
    struct ggml_backend_zdnn_context        * ctx     = (struct ggml_backend_zdnn_context *)backend->context;
    // struct ggml_backend_zdnn_device_context * ctx_dev = (struct ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;

    for (int i = 0; i < gf->n_nodes; i++) {
        struct ggml_tensor * node = gf->nodes[i];

        if (ggml_is_empty(node)
            || node->op == GGML_OP_NONE
            || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE
            || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        // #ifndef NDEBUG
        // assert(node->buffer->buft == ggml_backend_zdnn_buffer_type());
        // for (int j = 0; j < GGML_MAX_SRC; j++) {
        //     if (node->src[j] != nullptr) {
        //         assert(node->src[j]->buffer);
        //         assert(node->src[j]->buffer->buft == ggml_backend_zdnn_buffer_type() ||
        //                ggml_backend_buft_is_host(node->src[j]->buffer->buft));
        //     }
        // }
        // #endif  // NDEBUG

        bool ok = ggml_backend_zdnn_compute_forward(ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, ggml_op_name(node->op), node->name);
            return GGML_STATUS_FAILED;
        }

        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_zdnn_init_bias_tensor(struct ggml_backend_zdnn_buffer * buffer, struct ggml_tensor * tensor) {
    zdnn_init_pre_transformed_desc(
        ZDNN_1D,
        FP32,
        &buffer->pre_tfm_desc,
        tensor->ne[0], 1, 1, 1
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&buffer->pre_tfm_desc, &buffer->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&buffer->pre_tfm_desc, &buffer->tfm_desc, &buffer->ztensor));
}

static void ggml_zdnn_init_tensor(struct ggml_backend_zdnn_buffer * buffer, struct ggml_tensor * tensor) {
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    FP32,
                    &buffer->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } break;
        default:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_NCHW,
                    FP32,
                    &buffer->pre_tfm_desc,
                    tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
                );
            } break;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&buffer->pre_tfm_desc, &buffer->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&buffer->pre_tfm_desc, &buffer->tfm_desc, &buffer->ztensor));
}

////////////////////////////////////////////////////////////////////////////////

//
// backend interface
//

static void ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    struct ggml_backend_zdnn_buffer_context * ctx = (struct ggml_backend_zdnn_buffer_context *)buffer->context;

    for (int i = 0; i < ctx->n_buffers; i++) {
        struct ggml_backend_zdnn_buffer * buf = &ctx->buffers[i];

        // free any extra buffers (e.g., bias)
        if (buf->extra != nullptr) {
            zdnn_free_ztensor_buffer(&buf->extra->ztensor);
            free(buf->extra->data);
        }
        zdnn_free_ztensor_buffer(&buf->ztensor);
    }

    if (ctx->owned) {
        free(ctx->all_data);
    }

    free(ctx);
}

static void * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    struct ggml_backend_zdnn_buffer_context * ctx = (struct ggml_backend_zdnn_buffer_context *)buffer->context;
    return ctx->all_data;
}

static enum ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    struct ggml_backend_zdnn_buffer_context * ctx = (struct ggml_backend_zdnn_buffer_context *)buffer->context;

    // Create a dedicated buffer entry for this tensor
    int tensor_buffer_idx;
    int bias_buffer_idx;
    const int64_t tsize = ggml_nbytes(tensor);

    struct ggml_backend_zdnn_buffer * tensor_buffer;
    tensor_buffer_idx   = ctx->n_buffers;
    tensor_buffer       = &ctx->buffers[tensor_buffer_idx];
    tensor_buffer->data = tensor->data;
    tensor_buffer->size = tsize;
    snprintf(tensor_buffer->name, sizeof(tensor_buffer->name), "%s", tensor->name);

    ggml_zdnn_init_tensor(tensor_buffer, tensor);
    ctx->n_buffers++;

    if (tensor->op == GGML_OP_MUL_MAT) {
        struct ggml_backend_zdnn_buffer * bias_buffer;
        bias_buffer_idx   = tensor_buffer_idx + 1;
        bias_buffer       = &ctx->buffers[bias_buffer_idx];
        bias_buffer->data = calloc(tensor->ne[0], tensor->ne[0] * sizeof(float));
        bias_buffer->size = tensor->ne[0] * sizeof(float);
        snprintf(bias_buffer->name, sizeof(bias_buffer->name), "%s.bias", tensor->name);

        ggml_zdnn_init_bias_tensor(bias_buffer, tensor);
        ctx->n_buffers++;

        tensor_buffer->extra = bias_buffer;

        GGML_LOG_INFO("%s: initialized bias tensor '%s' in buffer %d, size = %8.2f MiB\n",
                      __func__, bias_buffer->name, bias_buffer_idx,
                      (float)bias_buffer->size / (1024.0f * 1024.0f));
    }

    GGML_LOG_INFO("%s: initialized tensor '%s' in buffer %d, size = %8.2f MiB\n",
                  __func__, ctx->buffers[tensor_buffer_idx].name, tensor_buffer_idx,
                  (float)tsize / (1024.0f * 1024.0f));

    tensor->extra = &ctx->buffers[tensor_buffer_idx];

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;

    // Log only for MUL_MAT operations
    if (tensor->op == GGML_OP_MUL_MAT) {
        GGML_LOG_INFO("%s: MUL_MAT operation - tensor '%s', size = %zu bytes\n",
                      __func__, tensor->name, size);
        GGML_LOG_INFO("%s: tensor->extra->extra = %p\n",
                      __func__, extra->extra);
    }

    // if extra buffer exists, transform the ztensor with the buffer data. for e.g., bias
    if (extra->extra != nullptr) {
        GGML_LOG_INFO("%s: transforming bias ztensor for tensor '%s', bias size = %zu bytes\n",
                      __func__, tensor->name, extra->extra->size);

        zdnn_status status = zdnn_transform_ztensor(&extra->extra->ztensor, extra->extra->data);
        if (status != ZDNN_OK) {
            GGML_LOG_ERROR("%s: failed to transform bias ztensor for tensor '%s', status = %d\n",
                           __func__, tensor->name, status);
        } else {
            GGML_LOG_INFO("%s: successfully transformed bias ztensor for tensor '%s'\n",
                          __func__, tensor->name);
        }
        ZDNN_CHECK(status);
    }

    // for all other data
    ZDNN_CHECK(zdnn_transform_ztensor(&extra->ztensor, (void *)((char *)tensor->data + offset)));

    memcpy((char *)tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    ZDNN_CHECK(zdnn_transform_origtensor(&extra->ztensor, (void *)((char *)tensor->data + offset)));

    memcpy(data, (const char *)tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    struct ggml_backend_zdnn_buffer_context * ctx = (struct ggml_backend_zdnn_buffer_context *)buffer->context;
    memset(ctx->all_data, value, ctx->all_size);
}

static struct ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .cpy_tensor    = */ NULL,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ NULL,
};

//
// default buffer type
//

static const char * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_zdnn_buffer_context * ctx = (struct ggml_backend_zdnn_buffer_context *)calloc(1, sizeof(struct ggml_backend_zdnn_buffer_context));

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if (size_aligned % size_page != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    struct ggml_backend_zdnn_device_context * ctx_dev = (struct ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device != 0);
    int device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data = ggml_aligned_malloc(size_aligned);
    ctx->all_size = size_aligned;
    ctx->owned = true;
    ctx->n_buffers = 1;

    if (ctx->all_data != NULL) {
        ctx->buffers[0].data = ctx->all_data;
        ctx->buffers[0].size = size;
    }

    if (size_aligned > 0 && (ctx->all_data == NULL)) {
        GGML_LOG_ERROR("%s: failed to allocate buffer, size = %8.2f MiB\n",
                       __func__, (float)size_aligned / (1024.0f / 1024.0f));
        free(ctx);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

static size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 256;

    GGML_UNUSED(buft);
}

static bool ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_zdnn;
}

static const char * ggml_backend_zdnn_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "ZDNN_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_from_ptr_type_zdnn = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_zdnn_buffer_from_ptr_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_from_ptr_type_zdnn;
}

static const char * ggml_backend_zdnn_name(ggml_backend_t backend) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_zdnn_free(ggml_backend_t backend) {
    struct ggml_backend_zdnn_context * ctx = (struct ggml_backend_zdnn_context *)backend->context;

    ggml_aligned_free(ctx, 0);
    free(backend);
}

static enum ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return ggml_zdnn_graph_compute(backend, cgraph);
}

static struct ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name           = */ ggml_backend_zdnn_name,
    /* .free               = */ ggml_backend_zdnn_free,
    /* .set_tensor_async   = */ NULL,
    /* .get_tensor_async   = */ NULL,
    /* .cpy_tensor_async   = */ NULL,
    /* .synchronize        = */ NULL,
    /* .graph_plan_create  = */ NULL,
    /* .graph_plan_free    = */ NULL,
    /* .graph_plan_update  = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute      = */ ggml_backend_zdnn_graph_compute,
    /* .event_record       = */ NULL,
    /* .event_wait         = */ NULL,
};

static ggml_guid_t ggml_backend_zdnn_guid(void) {
    static const char * guid_str = "IBM-ZDNN_ACCELER";
    return reinterpret_cast<ggml_guid_t>((void *)guid_str);
}

ggml_backend_t ggml_backend_zdnn_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_zdnn_reg(), 0);

    struct ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend *)ggml_aligned_malloc(sizeof(struct ggml_backend));

    * backend = (struct ggml_backend) {
        /* .guid     = */ ggml_backend_zdnn_guid(),
        /* .iface    = */ ggml_backend_zdnn_i,
        /* .device   = */ dev,
        /* .context  = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_zdnn(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

static const char * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

static const char * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    return "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

static void ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 1;
    *total = 1;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = (struct ggml_backend_dev_caps) {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, const char * params) {
    struct ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend *)ggml_aligned_malloc(sizeof(struct ggml_backend));

    * backend = (struct ggml_backend) {
        /* .guid     = */ ggml_backend_zdnn_guid(),
        /* .iface    = */ ggml_backend_zdnn_i,
        /* .device   = */ dev,
        /* .context  = */ ctx,
    };

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_zdnn_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    struct ggml_backend_zdnn_buffer_context * ctx = (struct ggml_backend_zdnn_buffer_context *)calloc(1, sizeof(struct ggml_backend_zdnn_buffer_context));

    ctx->all_data = ptr;
    ctx->all_size = size;
    ctx->owned = false;
    ctx->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offset = (uintptr_t)ptr % size_page;
        ptr = (void *)((char *)ptr - offset);
        size += offset;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    struct ggml_backend_zdnn_device_context * ctx_dev = (struct ggml_backend_zdnn_device_context *)dev->context;

    GGML_ASSERT(ctx_dev->zdnn_device != 0);
    int device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->buffers[ctx->n_buffers].data = ptr;
    ctx->buffers[ctx->n_buffers].size = size;

    GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB\n",
                  __func__, (float)size_aligned / (1024.0f / 1024.0f));

    ++ctx->n_buffers;

    return ggml_backend_buffer_init(ggml_backend_zdnn_buffer_from_ptr_type(), ggml_backend_zdnn_buffer_i, ctx, size);

    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    struct ggml_backend_zdnn_device_context * ctx_dev = (struct ggml_backend_zdnn_device_context *)dev->context;

    return ggml_zdnn_supports_op(ctx_dev, op);
}

static bool ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name ||
        buft->iface.get_name == ggml_backend_zdnn_buffer_from_ptr_type_get_name;

    GGML_UNUSED(dev);
}

static bool ggml_backend_zdnn_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return false;

    GGML_UNUSED(dev);
    GGML_UNUSED(op);
}

static struct ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_zdnn_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ggml_backend_zdnn_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// backend registry
//

static const char * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_zdnn_reg_device_count(ggml_backend_reg_t reg) {
    if (!zdnn_is_nnpa_installed()) {
        return 0;
    }

    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    return &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    // Change once we have proper detections
    { "NNPA_PARMBLK", "1"},
};

static struct ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

static void * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static struct ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_count,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address,
};

static void ggml_zdnn_cleanup(void) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

ggml_backend_reg_t ggml_backend_zdnn_reg(void) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = (ggml_backend_reg) {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ NULL,
        };

        g_ggml_backend_zdnn_device = (ggml_backend_device) {
            /* .iface   = */ ggml_backend_zdnn_device_i,
            /* .reg     = */ &g_ggml_backend_zdnn_reg,
            /* .context = */ &g_ggml_ctx_dev_main,
        };
    }

    return &g_ggml_backend_zdnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
