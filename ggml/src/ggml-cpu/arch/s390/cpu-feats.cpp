#include "ggml-backend-impl.h"

#if defined(__s390x__)
#include <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT
#define HWCAP_VXRS_EXT (1 << 13)
#endif

#ifndef HWCAP_VXRS_EXT2
#define HWCAP_VXRS_EXT2 (1 << 15)
#endif

#ifndef HWCAP_NNPA
#define HWCAP_NNPA (1 << 20)
#endif

struct s390x_features {
    bool has_vxe  = false;
    bool has_vxe2 = false;
    bool has_nnpa = false;

    s390x_features() {
        uint32_t hwcap  = getauxval(AT_HWCAP);
        uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe  = !!(hwcap & HWCAP_VXRS_EXT);
        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);

        GGML_LOG_INFO("s390x features detected: VXE=%d, VXE2=%d, NNPA=%d",
                      has_vxe, has_vxe2, has_nnpa);
    }
};

static int ggml_backend_cpu_s390x_score() {
    int score = 1;
    s390x_features sf;

#ifdef GGML_USE_VXE
    if (!sf.has_vxe) { return 0; }
    score += 1 << 1;
#endif
#ifdef GGML_USE_VXE2
    if (!sf.has_vxe2) { return 0; }
    score += 1 << 2;
#endif
#ifdef GGML_USE_NNPA
    if (!sf.has_nnpa) { return 0; }
    score += 1 << 3;
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#endif  // __s390x__
