ARG GCC_VERSION=15.2.0
ARG DEBIAN_VERSION=12


FROM --platform=linux/s390x gcc:${GCC_VERSION} AS build

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        git cmake ccache ninja-build \
        python3 python3-pip python3-dev \
        libcurl4-openssl-dev libopenblas-openmp-dev && \
    rm -rf /var/lib/apt/lists/*

# Install rustc for pip installation
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

WORKDIR /app
COPY . .

RUN pip3 install --no-cache-dir --prefix=/gguf-py -r requirements.txt

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/build \
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
COPY --from=build /usr/lib/s390x-linux-gnu /lib


### Non-Hardened Base Target
FROM --platform=linux/s390x debian:${DEBIAN_VERSION}-slim AS base

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && \
    apt install -y libgomp1 curl && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

# Copy llama.cpp libraries
COPY --from=collector /llama.cpp/lib /usr/lib/s390x-linux-gnu

# Copy all distro libraries
COPY --from=collector /lib /lib/s390x-linux-gnu


### Full
FROM base AS full

USER root:root
WORKDIR /app

# Fix rustc not found
ENV PATH="/root/.cargo/bin:${PATH}"

COPY --from=collector /llama.cpp/bin /app
COPY --from=collector /llama.cpp/gguf-py /app/gguf-py

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && \
    apt install -y --no-install-recommends \
        git python3 python3-pip python3-dev && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

RUN --mount=type=cache,target=/root/.cargo \
    curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    pip install -r /app/gguf-py/requirements.txt

ENTRYPOINT [ "/app/tools.sh" ]


### CLI Only
FROM --platform=linux/s390x base AS light

USER root:root
WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /llama.cpp/bin/llama-cli       /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/libggml-cpu.so  /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/libggml-blas.so /llama.cpp/bin

ENTRYPOINT [ "/llama.cpp/bin/llama-cli" ]


### Hardened Server
FROM --platform=linux/s390x gcr.io/distroless/cc-debian${DEBIAN_VERSION}:nonroot AS server

ENV LLAMA_ARG_HOST=0.0.0.0

USER nonroot:nonroot
WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /llama.cpp/bin/llama-server /llama.cpp/bin
COPY --from=collector /llama.cpp/lib /usr/lib/s390x-linux-gnu

# Fixes model loading errors
COPY --from=collector /llama.cpp/bin/libggml-cpu.so /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/libggml-blas.so /llama.cpp/bin

# Copy all distro libraries
COPY --from=collector /lib /lib/s390x-linux-gnu

EXPOSE 8080

ENTRYPOINT [ "/llama.cpp/bin/llama-server" ]
