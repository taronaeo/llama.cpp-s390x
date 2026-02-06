#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

#if defined(__VXE__) || defined(__VXE2__)
#endif  // __VXE__ || __VXE2__

