## GGML Backend Development Guide

## Table of Contents

- [Introduction](#introduction)
- [GGML Backend API](#ggml-backend-api)
    - [Backend Registration](#backend-registration)
        - [ggml_backend_custom_reg_get_name](#ggml_backend_custom_reg_get_name)
        - [ggml_backend_custom_reg_get_device_count](#ggml_backend_custom_reg_get_device_count)
        - [ggml_backend_custom_reg_get_device](#ggml_backend_custom_reg_get_device)
        - [ggml_backend_custom_get_proc_address](#ggml_backend_custom_get_proc_address)
    - [Backend Device Registration](#backend-device-registration)
        - [ggml_backend_custom_device_get_name](#ggml_backend_custom_device_get_name)
        - [ggml_backend_custom_device_get_description](#ggml_backend_custom_device_get_description)
        - [ggml_backend_custom_device_get_memory](#ggml_backend_custom_device_get_memory)
        - [ggml_backend_custom_device_get_type](#ggml_backend_custom_device_get_type)
        - [ggml_backend_custom_device_get_props](#ggml_backend_custom_device_get_props)
        - [ggml_backend_custom_device_init_backend](#ggml_backend_custom_device_init_backend)
        - [ggml_backend_custom_device_get_buffer_type](#ggml_backend_custom_device_get_buffer_type)
        - [ggml_backend_custom_device_get_host_buffer_type](#ggml_backend_custom_device_get_host_buffer_type)
        - [ggml_backend_custom_device_get_buffer_from_host_ptr](#ggml_backend_custom_device_get_buffer_from_host_ptr)
        - [ggml_backend_custom_device_supports_op](#ggml_backend_custom_device_supports_op)
        - [ggml_backend_custom_device_supports_buft](#ggml_backend_custom_device_supports_buft)
        - [ggml_backend_custom_device_offload_op](#ggml_backend_custom_device_offload_op)
        - [ggml_backend_custom_device_event_new](#ggml_backend_custom_device_event_new)
        - [ggml_backend_custom_device_event_free](#ggml_backend_custom_device_event_free)
        - [ggml_backend_custom_device_event_synchronize](#ggml_backend_custom_device_event_synchronize)
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

Backend Registration Interface

<br />

#### ggml_backend_custom_reg_get_name

```c++
static const char * ggml_backend_custom_reg_get_name(ggml_backend_reg_t reg)
```

Return the name of the backend.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4734-L4737
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L658-L662
</details>

<br />

#### ggml_backend_custom_reg_get_device_count

```c++
static size_t ggml_backend_custom_reg_get_device_count(ggml_backend_reg_t reg)
```

Returns the total number of available devices.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4739-L4742
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L664-L668
</details>

<br />

#### ggml_backend_custom_reg_get_device

```c++
static ggml_backend_dev_t ggml_backend_custom_reg_get_device(ggml_backend_reg_t reg, size_t index)
```

Returns the GGML device interface via an index.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4744-L4748
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L670-L677
</details>

<br />

#### ggml_backend_custom_get_proc_address

```c++
static void * ggml_backend_custom_get_proc_address(ggml_backend_reg_t reg, const char * name)
```

OPTIONAL - Registers pointers to custom functions in the backend

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4811-L4826
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L692-L700
</details>

<br />

### Backend Device Registration

```c++
static const ggml_backend_device_i ggml_backend_custom_device_interface = {
    /* .get_name                = */ ggml_backend_custom_device_get_name,
    /* .get_description         = */ ggml_backend_custom_device_get_description,
    /* .get_memory              = */ ggml_backend_custom_device_get_memory,
    /* .get_type                = */ ggml_backend_custom_device_get_type,
    /* .get_props               = */ ggml_backend_custom_device_get_props,
    /* .init_backend            = */ ggml_backend_custom_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_custom_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_custom_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_custom_device_get_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_custom_device_supports_op,
    /* .supports_buft           = */ ggml_backend_custom_device_supports_buft,
    /* .offload_op              = */ ggml_backend_custom_device_offload_op,
    /* .event_new               = */ ggml_backend_custom_device_event_new,
    /* .event_free              = */ ggml_backend_custom_device_event_free,
    /* .event_synchronize       = */ ggml_backend_custom_device_event_synchronize,
};
```

Device Registration Interface

#### ggml_backend_custom_device_get_name

```c++
static const char * ggml_backend_custom_device_get_name(ggml_backend_dev_t dev)
```

Returns the name of the device. It should be in the form of `device name[device number]` where `[]` is optional.

For example, for backends with multiple devices: CUDA0, CUDA1. For backends with single devices: Metal.

**NOTE**: The device name will be used in user-specified flags such as `-ot` to override tensors to a specific device.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4122-L4125
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L521-L525
</details>

<br />

#### ggml_backend_custom_device_get_description

```c++
static const char * ggml_backend_custom_device_get_description(ggml_backend_dev_t dev)
```

Returns the description of the device. You can return the device model here.

For example, NVIDIA RTX 2060.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4127-L4130
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L527-L531
</details>

<br />

#### ggml_backend_custom_device_get_memory

```c++
static void ggml_backend_custom_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total)
```

Returns the available and total memory of the device.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4208-L4236
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L533-L537
</details>

<br />

#### ggml_backend_custom_device_get_type

```c++
static enum ggml_backend_dev_type ggml_backend_custom_device_get_type(ggml_backend_dev_t dev)
```

Returns the device type. Choose a type from the following:

```c++
enum ggml_backend_dev_type {
    // CPU device using system memory
    GGML_BACKEND_DEVICE_TYPE_CPU,
    // GPU device using dedicated memory
    GGML_BACKEND_DEVICE_TYPE_GPU,
    // integrated GPU device using host memory
    GGML_BACKEND_DEVICE_TYPE_IGPU,
    // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
    GGML_BACKEND_DEVICE_TYPE_ACCEL
};
```

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4238-L4241
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L539-L543
</details>

<br />

#### ggml_backend_custom_device_get_props

```c++
static void ggml_backend_custom_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props)
```

Returns the device properties. Use this function to report information on the device capabilities.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4243-L4265
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L545-L558
</details>

<br />

#### ggml_backend_custom_device_init_backend

```c++
static ggml_backend_t ggml_backend_custom_device_init_backend(ggml_backend_dev_t dev, const char * params)
```

Returns an initialized backend of the device.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4267-L4271
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L560-L583
</details>

<br />

#### ggml_backend_custom_device_get_buffer_type

```c++

static ggml_backend_buffer_type_t ggml_backend_custom_device_get_buffer_type(ggml_backend_dev_t dev)
```

Returns the preferred buffer type for the device.

If you do not want to manage the buffer, you can return `ggml_backend_cpu_buffer_type()` instead.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4273-L4276
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L585-L591
</details>

<br />

#### ggml_backend_custom_device_get_host_buffer_type

```c++
static ggml_backend_buffer_type_t ggml_backend_custom_device_get_host_buffer_type(ggml_backend_dev_t dev)
```

OPTIONAL - TODO -

<br />

#### ggml_backend_custom_device_get_buffer_from_host_ptr

```c++
static ggml_backend_buffer_t ggml_backend_custom_device_get_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size)
```

OPTIONAL - TODO -

<br />

#### ggml_backend_custom_device_supports_op

```c++
static bool ggml_backend_custom_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * dst)
```

Returns the status of operation support for the device.

The GGML scheduler will use this function to check for operation support. If the operation is declared to be supported on the device, the GGML scheduler will allocate the tensor buffer to the device backend.

> [!IMPORTANT]
> NOTE 1: If an operation is declared to be supported, you will need to update the `ggml_backend_custom_graph_compute` function and implement support for it.

> [!IMPORTANT]
> NOTE 2: Do not attempt to reject tensors like this
> ```c++
> if (dst->ne[1] == 1) {
>   return false;
> }
> ```
> This will cause the GGML scheduler to unexpectedly reallocate the tensor buffer to a backend that is able to support it and will be bad for performance.

<details>
<summary>CUDA Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-cuda/ggml-cuda.cu#L4284-L4649
</details>

<details>
<summary>Metal Code Example</summary>

https://github.com/ggml-org/llama.cpp/blob/0db81098494023775a704a44042c317d36c91f24/ggml/src/ggml-metal/ggml-metal.cpp#L601-L605
</details>

<br />

#### ggml_backend_custom_device_supports_buft

```c++
static bool ggml_backend_custom_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft)
```

<br />

#### ggml_backend_custom_device_offload_op

```c++
static bool ggml_backend_custom_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op)
```

OPTIONAL -

<br />

#### ggml_backend_custom_device_event_new

```c++
static ggml_backend_event_t ggml_backend_custom_device_event_new(ggml_backend_dev_t dev)
```

OPTIONAL -

<br />

#### ggml_backend_custom_device_event_free

```c++
static void ggml_backend_custom_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event)
```

OPTIONAL -

<br />

#### ggml_backend_custom_device_event_synchronize

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



