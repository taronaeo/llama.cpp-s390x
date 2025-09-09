ARG GCC_VERSION=15.2.0
ARG DEBIAN_VERSION=12
ARG BUILD_DIR=/app

FROM gcc:${GCC_VERSION} AS build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        git cmake ccache ninja-build \
        libcurl4-openssl-dev libopenblas-openmp-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${BUILD_DIR}
COPY . .

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=${BUILD_DIR}/build \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS && \
    cmake --build build --config Release -j $(nproc) && \
    cmake --install build --prefix /opt/llama.cpp

# TODO: DOUBLE CHECK ALL FILES ARE COPIED INTO COLLECTOR
# TODO: Switch to COPY operation instead for caching
RUN cp *.py /opt/llama.cpp \
    && cp -r gguf-py /opt/llama.cpp \
    && cp -r requirements /opt/llama.cpp \
    && cp requirements.txt /opt/llama.cpp \
    && cp .devops/tools.sh /opt/llama.cpp/tools.sh

RUN ls -laR /opt/llama.cpp


### Collect all llama.cpp binaries, libraries and distro libraries
FROM --platform=linux/s390x scratch AS collector

# Copy llama.cpp binaries and libraries
COPY --from=build /opt/llama.cpp/bin /bin/llama.cpp
COPY --from=build /opt/llama.cpp/lib /lib/llama.cpp

# Copy all shared libraries from distro
COPY --from=build /usr/lib/s390x-linux-gnu /lib/distro


### Non-Hardened Base Target
FROM --platform=linux/s390x debian:${DEBIAN_VERSION}-slim AS base

RUN apt update -y \
    && apt install -y libgomp1 curl \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

# Copy llama.cpp libraries
COPY --from=collector /lib/llama.cpp /usr/lib/s390x-linux-gnu

# Copy all distro libraries
COPY --from=collector /lib/distro /lib/s390x-linux-gnu


### CLI Only
FROM --platform=linux/s390x base AS light

USER root:root
WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /bin/llama.cpp/llama-cli       /llama.cpp/bin
COPY --from=collector /bin/llama.cpp/libggml-cpu.so  /llama.cpp/bin
COPY --from=collector /bin/llama.cpp/libggml-blas.so /llama.cpp/bin

ENTRYPOINT [ "/llama.cpp/bin/llama-cli" ]


### Hardened Server
FROM --platform=linux/s390x gcr.io/distroless/cc-debian${DEBIAN_VERSION}:nonroot AS server

ENV LLAMA_ARG_HOST=0.0.0.0

USER nonroot:nonroot
WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /bin/llama.cpp/llama-server /llama.cpp/bin
COPY --from=collector /lib/llama.cpp /usr/lib/s390x-linux-gnu

# Fixes model loading errors
COPY --from=collector /bin/llama.cpp/libggml-cpu.so /llama.cpp/bin
COPY --from=collector /bin/llama.cpp/libggml-blas.so /llama.cpp/bin

# Copy all distro libraries
COPY --from=collector /lib/distro /lib/s390x-linux-gnu

EXPOSE 8080

ENTRYPOINT [ "/llama.cpp/bin/llama-server" ]
