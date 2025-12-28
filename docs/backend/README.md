## GGML Backend Development Guide

## Table of Contents

- [Introduction](#introduction)
- [GGML Backend API](#ggml-backend-api)
    - [Backend Registration](#backend-registration)
- [Backend Structure](#backend-structure)
- [Backend Components](#backend-components)
    - [Registering Custom Backend](#registering-custom-backend)
    - [Registering Custom Backend Device](#registering-custom-backend-device)
    - [Registering Custom Backend Tensor Operations](#registering-custom-backend-tensor-operations)
    - [Registering Custom Backend Buffer](#registering-custom-backend-buffer)
    - [Registering Custom Backend Buffer Type](#registering-custom-backend-buffer-type)

---

## Introduction

The GGML backend provide abstraction layers that enable the GGML tensor library to run on different hardware platforms and compute devices. It provides a unified interface for backend initialization, buffer (memory) management, and compute graph optimization across various hardware.

This guidebook serves as a point-of-reference documentation for implementing and debugging custom backends for GGML/Llama.cpp and also provides notes on common misconceptions that developers have.

---

## GGML Backend API

**NOTE**: This section will use the backend name of `custom` as an example. You should replace this with the actual backend name.

### Backend Registration

```c++
static const ggml_backend_reg_i ggml_backend_custom_reg_i = {
    /* .get_name         = */ ggml_backend_custom_reg_get_name,
    /* .get_device_count = */ ggml_backend_custom_reg_get_device_count,
    /* .get_device       = */ ggml_backend_custom_reg_get_device,
    /* .get_proc_address = */ ggml_backend_custom_get_proc_address,
};
```

<br />

```c++
static const char * ggml_backend_custom_reg_get_name(ggml_backend_reg_t reg)
```

Return the name of the backend.

<details>
<summary>CUDA Code Example</summary>

```c++
static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;  // CUDA
}
```
</details>

<details>
<summary>Metal Code Example</summary>

```c++
static const char * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    return "Metal";

    GGML_UNUSED(reg);
}
```
</details>

<br />

```c++
static size_t ggml_backend_custom_reg_get_device_count(ggml_backend_reg_t reg)
```

Returns the total number of available devices.

<details>
<summary>CUDA Code Example</summary>

```c++
static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}
```
</details>

<details>
<summary>Metal Code Example</summary>

```c++
static size_t ggml_backend_metal_reg_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}
```
</details>

<br />

```c++
static ggml_backend_dev_t ggml_backend_custom_reg_get_device(ggml_backend_reg_t reg, size_t index)
```

Returns the GGML device interface via an index.

<details>
<summary>CUDA Code Example</summary>

```c++
static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}
```
</details>

<details>
<summary>Metal Code Example</summary>

```c++
static ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    return &g_ggml_metal_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}
```
</details>

<br />

```c++
static void * ggml_backend_custom_get_proc_address(ggml_backend_reg_t reg, const char * name)
```

OPTIONAL - Registers pointers to custom functions in the backend

<details>
<summary>CPU Code Example</summary>

```c++
static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}
```
</details>

<details>
<summary>CUDA Code Example</summary>

```c++
static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_PEER_MAX_BATCH_SIZE
        features.push_back({ "PEER_MAX_BATCH_SIZE", STRINGIFY(GGML_CUDA_PEER_MAX_BATCH_SIZE) });
    #endif

    #ifdef GGML_CUDA_FA_ALL_QUANTS
        features.push_back({ "FA_ALL_QUANTS", "1" });
    #endif

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_cuda_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}
```
</details>

<br />

### Backend Device Registration

```c++
static const char * ggml_backend_custom_device_get_name(ggml_backend_dev_t dev)
```

Returns the name of the device. It should be in the form of `device name[device number]` where `[]` is optional.

For example, for backends with multiple devices: CUDA0, CUDA1. For backends with single devices: Metal.

**NOTE**: The device name will be used in user-specified flags such as `-ot` to override tensors to a specific device.

<details>
<summary>CUDA Code Example</summary>

```c++
static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}
```
</details>

<details>
<summary>Metal Code Example</summary>

```c++
static const char * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    return "Metal";

    GGML_UNUSED(dev);
}
```
</details>

<br />

```c++
static const char * ggml_backend_custom_device_get_description(ggml_backend_dev_t dev)
```

Returns the description of the device. You can return the device model here.

For example, NVIDIA RTX 2060.

<details>
<summary>CUDA Code Example</summary>

```c++
static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}
```
</details>

<details>
<summary>Metal Code Example</summary>

```c++
static const char * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_get_props(ctx_dev)->name;
}
```
</details>

<br />

```c++
static void ggml_backend_custom_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total)
```

<br />

```c++
static enum ggml_backend_dev_type ggml_backend_custom_device_get_type(ggml_backend_dev_t dev)
```

<br />

```c++
static void ggml_backend_custom_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props)
```

<br />

```c++
static ggml_backend_t ggml_backend_custom_device_init_backend(ggml_backend_dev_t dev, const char * params)
```

<br />

```c++

static ggml_backend_buffer_type_t ggml_backend_custom_device_get_buffer_type(ggml_backend_dev_t dev)
```

<br />

```c++
static ggml_backend_buffer_type_t ggml_backend_custom_device_get_host_buffer_type(ggml_backend_dev_t dev)
```

OPTIONAL -

<br />

```c++
static ggml_backend_buffer_t ggml_backend_custom_device_get_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size)
```

OPTIONAL -

<br />

```c++
static bool ggml_backend_custom_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * dst)
```

<br />

```c++
static bool ggml_backend_custom_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft)
```

<br />

```c++
static bool ggml_backend_custom_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op)
```

OPTIONAL -

<br />

```c++
static ggml_backend_event_t ggml_backend_custom_device_event_new(ggml_backend_dev_t dev)
```

OPTIONAL -

<br />

```c++
static void ggml_backend_custom_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event)
```

OPTIONAL -

<br />

```c++
static void ggml_backend_custom_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event)
```

OPTIONAL -

<br />

## Backend Structure

Every GGML custom backend consists of the following structure:

1. Backend Registration (`ggml_backend_reg_t`)
2. Backend Device Registration (`ggml_backend_dev_t`)
3. Backend Tensor Operations (`ggml_backend_t`)
4. Backend Buffer Registration (`ggml_backend_buffer_t`)
5. Backend Buffer Type Registration (`ggml_backend_buffer_type_t`)

## Backend Components

### Registering Custom Backend

You can register your backend using the `GGML_BACKEND_DL_IMPL` macro, for example,

```c++
struct ggml_backend_custom_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const ggml_backend_reg_i ggml_backend_custom_reg_i = {
    /* .get_name         = */ ggml_backend_custom_reg_get_name,
    /* .get_device_count = */ ggml_backend_custom_reg_get_device_count,
    /* .get_device       = */ ggml_backend_custom_reg_get_device,
    /* .get_proc_address = */ ggml_backend_custom_get_proc_address,
};

ggml_backend_reg_t ggml_backend_custom_reg(void) {
    ggml_backend_custom_reg_context * ctx = new ggml_backend_custom_reg_context;

    /*
     * If your backend involves multiple devices, it would be good to
     * discover and track all available devices here.
     */

    static ggml_backend_reg ggml_backend_custom_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_custom_reg_i,
        /* .context     = */ ctx,
    };

    return &ggml_backend_custom_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_custom_reg)
```

If your backend relies on a library (e.g., CUDA), you may wish to replace `GGML_BACKEND_API_VERSION` with the version of the library.

**Breakdown**:
- `.get_name`: Name of the custom backend (e.g., "CPU", "CUDA", "Metal", etc.)
- `.get_device_count`: Number of available devices recognised by the custom backend
- `.get_device`: Get device by index
- `.get_proc_address`: Custom function pointers (e.g., Get number of user-specified threads)

### Registering Custom Backend Device

Your backend devices should have been initialized during backend registration and tracked using a context

If your backend involves multiple devices, these devices should already

```c++
static const ggml_backend_device_i ggml_backend_custom_device_i = {
    /* .get_name             = */ ggml_backend_custom_device_get_name,
    /* .get_description      = */ ggml_backend_custom_device_get_description,
    /* .get_memory           = */ ggml_backend_custom_device_get_memory,
    /* .get_type             = */ ggml_backend_custom_device_get_type,
    /* .get_props            = */ ggml_backend_custom_device_get_props,
    /* .init_backend         = */ ggml_backend_custom_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_custom_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_custom_device_supports_op,
    /* .supports_buft        = */ ggml_backend_custom_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static ggml_backend_dev_t ggml_backend_custom_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    static ggml_backend_device ggml_backend_custom_device = {
        /* .iface   = */ ggml_backend_custom_device_i,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    return &ggml_backend_custom_device;
}
```

### Registering Custom Backend Tensor Operations

### Registering Custom Backend Buffer

### Registering Custom Backend Buffer Type


