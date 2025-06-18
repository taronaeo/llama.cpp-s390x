> [!IMPORTANT]
> This build documentation is specific only to IBM Z & LinuxONE mainframes (s390x). You can find the build documentation for other architectures: [build.md](build.md).

# Build llama.cpp locally (for s390x)

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

## CPU Build with BLAS

Building llama.cpp with BLAS support is highly recommended as it has shown to provide performance improvements.

```bash
cmake -S . -B build             \
    -DCMAKE_BUILD_TYPE=Release  \
    -DGGML_BLAS=ON              \
    -DGGML_BLAS_VENDOR=OpenBLAS

cmake --build build --config Release -j $(nproc)
```

**Notes**:
- For faster repeated compilation, install [ccache](https://ccache.dev/)
- By default, VXE/VXE2 is enabled. To disable it (not recommended):

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Release  \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DGGML_VXE=OFF
    
    cmake --build build --config Release -j $(nproc)
    ```

- For debug builds:

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Debug    \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS

    cmake --build build --config Debug -j $(nproc)
    ```

- For static builds, add `-DBUILD_SHARED_LIBS=OFF`:

    ```bash
    cmake -S . -B build             \
        -DCMAKE_BUILD_TYPE=Release  \
        -DGGML_BLAS=ON              \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DBUILD_SHARED_LIBS=OFF

    cmake --build build --config Release -j $(nproc)
    ```

## Getting GGUF Models

All models need to be converted to Big-Endian. You can achieve this in three cases:

1. Use pre-converted models verified for use on IBM Z & LinuxONE (easiest)

    You can find popular models pre-converted and verified at [s390x Ready Models](hf.co/collections/taronaeo/s390x-ready-models-672765393af438d0ccb72a08).

    These models and their respective tokenizers are verified to run correctly on IBM Z & LinuxONE.

2. Convert safetensors model to GGUF Big-Endian directly (recommended)

    ```bash
    python3 convert_hf_to_gguf.py \
        --outfile model-name-be.f16.gguf \
        --outtype f16 \
        --bigendian \
        model-directory/
    ```

    For example,
    
    ```bash
    python3 convert_hf_to_gguf.py \
        --outfile granite-3.3-2b-instruct-be.f16.gguf \
        --outtype f16 \
        --bigendian \
        granite-3.3-2b-instruct/
    ```

3. Convert existing GGUF Little-Endian model to Big-Endian

    ```bash
    python3 gguf-py/gguf/scripts/gguf_convert_endian.py model-name.f16.gguf BIG
    ```
    
    For example,
    ```bash
    python3 gguf-py/gguf/scripts/gguf_convert_endian.py granite-3.3-2b-instruct-le.f16.gguf BIG
    mv granite-3.3-2b-instruct-le.f16.gguf granite-3.3-2b-instruct-be.f16.gguf
    ```
    
    **Notes:**
    - The GGUF endian conversion script may not support all data types at the moment and may fail for some models/quantizations. When that happens, please try manually converting the safetensors model to GGUF Big-Endian via Step 2.



