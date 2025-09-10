ARG GCC_VERSION=15.2.0
ARG UBUNTU_VERSION=24.04


FROM --platform=linux/s390x gcc:${GCC_VERSION} AS build

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        git cmake ccache ninja-build \
        libcurl4-openssl-dev libopenblas-openmp-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/build \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_BACKEND_DL=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS && \
    cmake --build build --config Release -j $(nproc) && \
    cmake --install build --prefix /opt/llama.cpp

# TODO: DOUBLE CHECK ALL FILES ARE COPIED INTO COLLECTOR
COPY *.py             /opt/llama.cpp/bin
COPY .devops/tools.sh /opt/llama.cpp/bin

COPY gguf-py          /opt/llama.cpp/gguf-py
COPY requirements.txt /opt/llama.cpp/gguf-py
COPY requirements     /opt/llama.cpp/gguf-py/requirements

RUN ls -laR /opt/llama.cpp


### Collect all llama.cpp binaries, libraries and distro libraries
FROM --platform=linux/s390x scratch AS collector

# Copy llama.cpp binaries and libraries
COPY --from=build /opt/llama.cpp/bin /llama.cpp/bin
COPY --from=build /opt/llama.cpp/lib /llama.cpp/lib
COPY --from=build /opt/llama.cpp/gguf-py /llama.cpp/gguf-py

# Copy all shared libraries from distro
# COPY --from=build /usr/lib/s390x-linux-gnu /lib


### Base image
FROM --platform=linux/s390x ubuntu:${UBUNTU_VERSION} AS base

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && \
    apt install -y curl libgomp1 libopenblas-dev && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

# Copy llama.cpp libraries
COPY --from=collector /llama.cpp/lib /usr/lib/s390x-linux-gnu

# Copy all distro libraries
# COPY --from=collector /lib /lib/s390x-linux-gnu


### CLI Only
FROM --platform=linux/s390x base AS light

WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /llama.cpp/bin/llama-cli       /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/libggml-cpu.so  /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/libggml-blas.so /llama.cpp/bin

ENTRYPOINT [ "/llama.cpp/bin/llama-cli" ]


### Hardened Server
FROM --platform=linux/s390x base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /llama.cpp/bin/llama-server /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/libggml-cpu.so /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/libggml-blas.so /llama.cpp/bin

EXPOSE 8080

ENTRYPOINT [ "/llama.cpp/bin/llama-server" ]
