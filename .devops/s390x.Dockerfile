ARG GCC_VERSION=15.2.0
ARG UBUNTU_VERSION=24.10

FROM gcc:${GCC_VERSION} AS build

RUN apt update && \
    apt upgrade -y && \
    apt install -y git cmake libcurl4-openssl-dev libopenblas-openmp-dev

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

RUN cp *.py /opt/llama.cpp \
    && cp -r gguf-py /opt/llama.cpp \
    && cp -r requirements /opt/llama.cpp \
    && cp requirements.txt /opt/llama.cpp \
    && cp .devops/tools.sh /opt/llama.cpp/tools.sh

RUN ls -laR /opt/llama.cpp

FROM --platform=linux/s390x scratch AS collector

# Copy llama.cpp binaries and libraries
COPY --from=build /opt/llama.cpp/bin /bin/llama.cpp
COPY --from=build /opt/llama.cpp/lib /lib/llama.cpp

# Copy all shared libraries from distro
COPY --from=build /usr/lib/s390x-linux-gnu /lib/distro

FROM --platform=linux/s390x gcr.io/distroless/cc-debian12:nonroot AS server

ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080

# Copy llama.cpp binaries and libraries
COPY --from=collector /bin/llama.cpp/llama-server /
COPY --from=collector /lib/llama.cpp /usr/lib/s390x-linux-gnu

# Copy all shared libraries
COPY --from=collector /lib/distro /lib/s390x-linux-gnu

USER nonroot:nonroot
WORKDIR /models
EXPOSE ${LLAMA_ARG_PORT}

ENTRYPOINT [ "/llama-server" ]
