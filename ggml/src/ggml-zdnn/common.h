#pragma once

#ifndef ZDNN_COMMON_H
#define ZDNN_COMMON_H

#include "zdnn.h"
#include "../include/ggml.h"
#include "../include/ggml-zdnn.h"

#include <memory>

struct ggml_backend_zdnn_context {
  std::unique_ptr<char[]> work_data;
  size_t work_size = 0;
};

static zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);
void ggml_zdnn_op_add(ggml_backend_zdnn_context & ctx, ggml_tensor * tensor);

#endif /* ZDNN_COMMON_H */
