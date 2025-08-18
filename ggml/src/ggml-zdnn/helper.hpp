
#ifndef GGML_ZDNN_HELPER_HPP
#define GGML_ZDNN_HELPER_HPP

#include "zdnn.h"
#include "ggml-zdnn.h"
#include "ggml-zdnn-impl.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

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

[[deprecated("This function is deprecated, use ggml_zdnn_init_tensor instead.")]]
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

#endif // GGML_ZDNN_HELPER_HPP
