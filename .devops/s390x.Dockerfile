ARG GCC_VERSION=15.2.0
ARG UBUNTU_VERSION=24.10

FROM gcc:${GCC_VERSION} AS build

RUN apt-get update && \
    apt-get install -y git cmake libcurl4-openssl-dev libopenblas-openmp-dev

WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/root/.ccache \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
    && cmake --build build --config Release -j $(nproc) \
    && cmake --install build --prefix /opt/llama.cpp

RUN ls -la /opt/llama.cpp
