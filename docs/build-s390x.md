# Build llama.cpp locally (for s390x)

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](include/llama.h).

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
- For debug builds:

    ```bash
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Debug \
        -DGGML_BLAS=ON \
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


