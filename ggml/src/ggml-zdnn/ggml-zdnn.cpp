#include "zdnn.h"
#include "ggml-zdnn.h"
#include "ggml-zdnn-impl.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <csignal>
#include <unistd.h>

struct zdnn_extra {
    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    struct zdnn_extra * extra;  // for bias, etc.
};

struct ggml_backend_zdnn_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
};

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

static void ggml_backend_zdnn_mul_mat(ggml_backend_zdnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

    const char * zdnn_layouts[] = {"ZDNN_1D", "ZDNN_2D", "ZDNN_2DS", "ZDNN_3D", "ZDNN_3DS", "ZDNN_ZRH", "ZDNN_4D", "ZDNN_4DS", "ZDNN_NHWC", "ZDNN_NCHW", "ZDNN_FICO", "ZDNN_HWCK", "ZDNN_BIDIR_ZRH", "ZDNN_BIDIR_FICO"};

    // Log tensor extra information if available
    if (weights->extra) {
        zdnn_extra * weights_extra = (zdnn_extra *)weights->extra;
        GGML_LOG_INFO("%s: weights               shape=[%ld, %ld, %ld, %ld] type=%s\n",
              __func__, weights->ne[0], weights->ne[1], weights->ne[2], weights->ne[3],
              ggml_type_name(weights->type));
        GGML_LOG_INFO("%s: weights->pre_tfm_desc shape=[%ld, %ld, %ld, %ld] layout=%s type=%d\n",
              __func__, weights_extra->pre_tfm_desc.dim1, weights_extra->pre_tfm_desc.dim2, weights_extra->pre_tfm_desc.dim3, weights_extra->pre_tfm_desc.dim4,
              zdnn_layouts[weights_extra->pre_tfm_desc.layout], weights_extra->pre_tfm_desc.type);
        GGML_LOG_INFO("%s: weights->tfm_desc     shape=[%ld, %ld, %ld, %ld] layout=%s type=%d\n",
              __func__, weights_extra->tfm_desc.dim1, weights_extra->tfm_desc.dim2, weights_extra->tfm_desc.dim3, weights_extra->tfm_desc.dim4,
              zdnn_layouts[weights_extra->tfm_desc.layout], weights_extra->tfm_desc.type);
        GGML_LOG_INFO("%s: weights->ztensor      buffer_size=%zu is_transformed=%d rec_scale=%f\n",
              __func__, weights_extra->ztensor.buffer_size, weights_extra->ztensor.is_transformed,
              weights_extra->ztensor.rec_scale);
    }
    if (inputs->extra) {
        zdnn_extra * inputs_extra = (zdnn_extra *)inputs->extra;
        GGML_LOG_INFO("%s: inputs                shape=[%ld, %ld, %ld, %ld] type=%s\n",
              __func__, inputs->ne[0], inputs->ne[1], inputs->ne[2], inputs->ne[3],
              ggml_type_name(inputs->type));
        GGML_LOG_INFO("%s: inputs->pre_tfm_desc  shape=[%ld, %ld, %ld, %ld] layout=%s type=%d\n",
              __func__, inputs_extra->pre_tfm_desc.dim1, inputs_extra->pre_tfm_desc.dim2, inputs_extra->pre_tfm_desc.dim3, inputs_extra->pre_tfm_desc.dim4,
              zdnn_layouts[inputs_extra->pre_tfm_desc.layout], inputs_extra->pre_tfm_desc.type);
        GGML_LOG_INFO("%s: inputs->tfm_desc      shape=[%ld, %ld, %ld, %ld] layout=%s type=%d\n",
              __func__, inputs_extra->tfm_desc.dim1, inputs_extra->tfm_desc.dim2, inputs_extra->tfm_desc.dim3, inputs_extra->tfm_desc.dim4,
              zdnn_layouts[inputs_extra->tfm_desc.layout], inputs_extra->tfm_desc.type);
        GGML_LOG_INFO("%s: inputs->ztensor       buffer_size=%zu is_transformed=%d rec_scale=%f\n",
              __func__, inputs_extra->ztensor.buffer_size, inputs_extra->ztensor.is_transformed,
              inputs_extra->ztensor.rec_scale);
    }

    if (output->extra) {
        zdnn_extra * output_extra = (zdnn_extra *)output->extra;
        GGML_LOG_INFO("%s: output                shape=[%ld, %ld, %ld, %ld] type=%s\n",
              __func__, output->ne[0], output->ne[1], output->ne[2], output->ne[3],
              ggml_type_name(output->type));
        GGML_LOG_INFO("%s: output->pre_tfm_desc  shape=[%ld, %ld, %ld, %ld] layout=%s type=%d\n",
              __func__, output_extra->pre_tfm_desc.dim1, output_extra->pre_tfm_desc.dim2, output_extra->pre_tfm_desc.dim3, output_extra->pre_tfm_desc.dim4,
              zdnn_layouts[output_extra->pre_tfm_desc.layout], output_extra->pre_tfm_desc.type);
        GGML_LOG_INFO("%s: output->tfm_desc      shape=[%ld, %ld, %ld, %ld] layout=%s type=%d\n",
              __func__, output_extra->tfm_desc.dim1, output_extra->tfm_desc.dim2, output_extra->tfm_desc.dim3, output_extra->tfm_desc.dim4,
              zdnn_layouts[output_extra->tfm_desc.layout], output_extra->tfm_desc.type);
        GGML_LOG_INFO("%s: output->ztensor       buffer_size=%zu is_transformed=%d rec_scale=%f\n",
              __func__, output_extra->ztensor.buffer_size, output_extra->ztensor.is_transformed,
              output_extra->ztensor.rec_scale);
    }

    std::raise(SIGINT);

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

    const int64_t inputs_dim [GGML_MAX_DIMS] = { 1, 1, inputs_cols, inputs_rows };
    const int64_t weights_dim[GGML_MAX_DIMS] = { 1, 1, weights_cols, weights_rows };
    const int64_t bias_dim   [GGML_MAX_DIMS] = { 1, 1, 1, output_cols };
    const int64_t output_dim [GGML_MAX_DIMS] = { 1, 1, output_cols, output_rows };

    ggml_zdnn_create_tensor(pre_tfm_desc_inputs,  tfm_desc_inputs,  ztensor_inputs,  src1, inputs_dim,  ZDNN_2D);
    ggml_zdnn_create_tensor(pre_tfm_desc_weights, tfm_desc_weights, ztensor_weights, src0, weights_dim, ZDNN_2D);
    ggml_zdnn_create_tensor(pre_tfm_desc_bias,    tfm_desc_bias,    ztensor_bias,    dst,  bias_dim,    ZDNN_1D);
    ggml_zdnn_create_tensor(pre_tfm_desc_output,  tfm_desc_output,  ztensor_output,  dst,  output_dim,  ZDNN_2D);

    const size_t weights_size = ggml_element_size(src0);

    void * bias_data = (void *)calloc(output_cols, sizeof(ggml_element_size(dst)));

    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor_weights, weights->data));
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor_inputs,  inputs->data));
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor_bias,    bias_data));

    ZDNN_CHECK(zdnn_matmul_transpose_op(&ztensor_inputs, &ztensor_weights, &ztensor_bias,
                                        false, true, MATMUL_OP_ADDITION, &ztensor_output));
    ZDNN_CHECK(zdnn_transform_origtensor(&ztensor_output, output->data));

    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_weights));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_inputs));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_bias));
    ZDNN_CHECK(zdnn_free_ztensor_buffer(&ztensor_output));

    free(bias_data);
}

static void ggml_backend_zdnn_mul_mat_dispatch(ggml_backend_zdnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx);

    bool use_mul_mat_vec =
        (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F16)
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
        ggml_backend_zdnn_mul_mat(ctx, src0, src1, dst);
    }
}

static bool ggml_backend_zdnn_compute_forward(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            ggml_backend_zdnn_mul_mat_dispatch(ctx, dst->src[0], dst->src[1], dst);
            break;

        default:
            return false;
    }

    return true;
}

static const char * ggml_backend_zdnn_get_name(ggml_backend_t backend) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    delete ctx;
    delete backend;
}

static ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
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

        bool ok = ggml_backend_zdnn_compute_forward(ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, node->name, ggml_op_name(node->op));
        }

        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name           = */ ggml_backend_zdnn_get_name,
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
    // guid spells out IBM-NNPA-ACCELER
    static ggml_guid guid = { 0x49, 0x42, 0x4D, 0x2D, 0x4E, 0x4E, 0x50, 0x41,
                              0x2D, 0x41, 0x43, 0x43, 0x45, 0x4C, 0x45, 0x52 };

    return &guid;
}

ggml_backend_t ggml_backend_zdnn_init(void) {
    ggml_backend_zdnn_context * ctx = new ggml_backend_zdnn_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_zdnn_guid(),
        /* .iface     = */ ggml_backend_zdnn_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_zdnn_reg(), 0),
        /* .context   = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_zdnn(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

void ggml_backend_zdnn_set_n_threads(ggml_backend_t backend_zdnn, int n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend_zdnn));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend_zdnn->context;
    ctx->n_threads = n_threads;
}

static const char * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    return GGML_ZDNN_NAME;
}

static const char * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    return GGML_ZDNN_NAME;

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

static void ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_zdnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_zdnn_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static void * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;
    if (data % 256 != 0) {
        data = GGML_PAD(data, 256);
    }

    return (void *)data;
}

static ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    zdnn_extra * extra = (zdnn_extra *)malloc(sizeof(zdnn_extra));
    const int64_t dims[GGML_MAX_DIMS] = { 1, 1, tensor->ne[0], tensor->ne[1] };

    zdnn_init_pre_transformed_desc(
        ZDNN_2D,
        ggml_zdnn_type_mapping(tensor->type),
        &extra->pre_tfm_desc,
        dims[3], dims[2], dims[1], dims[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&extra->pre_tfm_desc, &extra->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&extra->pre_tfm_desc, &extra->tfm_desc, &extra->ztensor));

    if (tensor->op == GGML_OP_MUL_MAT) {
        zdnn_extra * bias_extra = (zdnn_extra *)malloc(sizeof(zdnn_extra));
        const int64_t bias_dims[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };

        zdnn_init_pre_transformed_desc(
            ZDNN_1D,
            ggml_zdnn_type_mapping(tensor->type),
            &bias_extra->pre_tfm_desc,
            bias_dims[3], bias_dims[2], bias_dims[1], bias_dims[0]
        );
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_extra->pre_tfm_desc, &bias_extra->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias_extra->pre_tfm_desc, &bias_extra->tfm_desc, &bias_extra->ztensor));

        extra->extra = bias_extra;
    }

    tensor->extra = extra;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer     = */ ggml_backend_zdnn_buffer_free_buffer,  // zdnn buffers are not owned by the backend
    /* .get_base        = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_zdnn_buffer_clear,
    /* .reset           = */ NULL,
};

static const ggml_backend_buffer_i ggml_backend_zdnn_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL,  // ptr is not owned by the buffer
    /* .get_base        = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_zdnn_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, data, size);
}

static size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 256;

    GGML_UNUSED(buft);
}

static bool ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type ggml_backend_zdnn_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL,  // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    return &ggml_backend_zdnn_buffer_type;
}

static const char * ggml_backend_zdnn_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_ZDNN_NAME "_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_from_ptr_type(void) {
    static ggml_backend_buffer_type ggml_backend_zdnn_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_zdnn_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL,  // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    return &ggml_backend_zdnn_buffer_type;
}

static ggml_backend_buffer_t ggml_backend_zdnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT((uintptr_t)ptr % 256 == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_zdnn_buffer_from_ptr_type(), ggml_backend_zdnn_buffer_from_ptr_i, ptr, size);
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
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];

            const int64_t ne10 = src1->ne[0];

            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            const int64_t max_batch = zdnn_get_nnpa_max_dim_idx_size();

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 <= max_batch && ne1 <= max_batch && ne10 <= max_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

static ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_zdnn_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ NULL,
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

static size_t ggml_backend_zdnn_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

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
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_zdnn_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_get_device_count,
    /* .get_device       = */ ggml_backend_zdnn_reg_get_device,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address,
};

ggml_backend_reg_t ggml_backend_zdnn_reg(void) {
    static ggml_backend_reg ggml_backend_zdnn_reg = {
        /* .api_version = */ GGML_ZDNN_VERSION,
        /* .iface       = */ ggml_backend_zdnn_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_zdnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
