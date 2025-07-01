#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_ZDNN_BACKEND_NAME     "ZDNN"
#define GGML_ZDNN_BACKEND_VERSION  ZDNN_VERNUM
#define GGML_ZDNN_TENSOR_ALIGNMENT 4096

GGML_BACKEND_API ggml_backend_t ggml_backend_zdnn_init(void);
GGML_BACKEND_API bool ggml_backend_is_zdnn(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_zdnn_reg(void);

#ifdef __cplusplus
}
#endif
