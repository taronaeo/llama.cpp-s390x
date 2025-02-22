#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_ZDNN_BACKEND_NAME    "ZDNN"
#define GGML_ZDNN_BACKEND_VERSION ZDNN_VERNUM

GGML_BACKEND_API ggml_backend_t ggml_backend_zdnn_init();
GGML_BACKEND_API bool ggml_backend_is_zdnn();

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_zdnn_reg();

#ifdef __cplusplus
}
#endif
