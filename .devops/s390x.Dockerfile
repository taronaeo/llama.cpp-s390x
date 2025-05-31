# vim: filetype=dockerfile

ARG UBUNTU_VERSION=24.10

FROM --platform=linux/s390x ubuntu:$UBUNTU_VERSION AS base
RUN apt update -y \
    && apt upgrade -y \
    && apt install -y curl libgomp1 \
    && apt install -y libcurl4-openssl-dev libopenblas-openmp-dev

FROM --platform=linux/s390x base AS build
RUN apt install -y git cmake pkg-config build-essential

WORKDIR /app

COPY . .

RUN cmake -S . -B build         \
    -DCMAKE_BUILD_TYPE=Release  \
    -DLLAMA_BUILD_TESTS=OFF     \
    -DGGML_BLAS=ON              \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    && cmake --build build --config Release -j$(nproc)

FROM --platform=linux/s390x scratch AS archive
COPY --from=build build/bin/llama-* /llama/bin
COPY --from=build build/bin/lib*.so /llama/lib

FROM --platform=linux/s390x base AS server
ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=archive /llama/lib /app
COPY --from=archive /llama/bin/llama-server /app

WORKDIR /app

HEALTHCHECK CMD curl --fail http://localhost:8080/health

ENTRYPOINT ["/app/llama-server"]
