#ifndef GGML_ZDNN_ADD_HPP
#define GGML_ZDNN_ADD_HPP

#include "common.hpp"

void ggml_zdnn_add_f(
    const ggml_backend_zdnn_context * ctx,
    const               ggml_tensor * src0,
    const               ggml_tensor * src1,
                        ggml_tensor * dst);

#endif  // GGML_ZDNN_ADD_HPP
