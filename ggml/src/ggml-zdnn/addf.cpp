#include "ggml.h"
#include "addf.hpp"

void ggml_zdnn_add_f(
    const ggml_backend_zdnn_context * ctx,
    const               ggml_tensor * src0,
    const               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    ggml_backend_zdnn_buffer * src0_extra = (ggml_backend_zdnn_buffer *)src0->extra;
    ggml_backend_zdnn_buffer * src1_extra = (ggml_backend_zdnn_buffer *)src1->extra;
    ggml_backend_zdnn_buffer * dst_extra  = (ggml_backend_zdnn_buffer *)dst->extra;

    ZDNN_CHECK(zdnn_add(&src0_extra->ztensor, &src1_extra->ztensor, &dst_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&dst_extra->ztensor, dst->data));

    GGML_UNUSED(ctx);
}
