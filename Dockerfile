FROM ubuntu:22.04

ARG KROKO_LICENSE=OFF

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      python3 \
      python3-dev \
      python3-pip \
      wget \
      libssl-dev \
      pkg-config \
      unzip \
      && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /workspace/sherpa-onnx
COPY . .

RUN mkdir build && cd build && \
    cmake \
      -D CMAKE_BUILD_TYPE=Release \
      -D SHERPA_ONNX_ENABLE_GPU=OFF \
      -D BUILD_SHARED_LIBS=ON \
      -D CMAKE_INSTALL_PREFIX=/opt/sherpa-onnx \
      -D KROKO_LICENSE=${KROKO_LICENSE} \
      -D CMAKE_CXX_FLAGS="-I/workspace/sherpa-onnx" \
      .. && \
    make -j$(nproc) && \
    make install && cd ..

RUN pip install .

ENV ORT_VERSION=1.17.1

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz && \
    mv onnxruntime-linux-x64-${ORT_VERSION}/lib/* /lib && \
    rm onnxruntime-linux-x64-${ORT_VERSION}.tgz

ENV LD_LIBRARY_PATH="/opt"

CMD ["/bin/bash"]
