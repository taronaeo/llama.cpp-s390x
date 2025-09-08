# vim: filetype=dockerfile

ARG GCC_VERSION=15.2.0
ARG UBUNTU_VERSION=24.10
ARG SUPPORT_ZDNN=OFF

FROM --platform=linux/s390x gcc:${GCC_VERSION} AS build
RUN apt update -y \
    && apt upgrade -y \
    && apt install -y --no-install-recommends cmake \
    && apt install -y --no-install-recommends libopenblas-openmp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN --mount=type=cache,target=/root/.ccache \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DGGML_ZDNN=$SUPPORT_ZDNN \
    && cmake --build build --config Release -j $(nproc)

RUN cmake --install build --prefix /opt/llama.cpp

FROM --platform=linux/s390x scratch AS llama-server

COPY --from=build /opt/llama.cpp/bin/llama-server /

WORKDIR /models
EXPOSE 8080

ENTRYPOINT ["/llama-server"]
