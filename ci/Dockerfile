FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       git g++ cmake python3 \
       libblas-dev \
       liblapack-dev \
       liblapacke-dev \
       libopenmpi-dev \
       openmpi-bin \
       libeigen3-dev \
       libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*