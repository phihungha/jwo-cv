ARG CUDA_VER=11.8.0
ARG CUDNN_VER=8
ARG UBUNTU_VER=22.04

FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-devel-ubuntu${UBUNTU_VER}
ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VER=3.11
RUN apt-get -q update \
    && apt-get -qy install python${PYTHON_VER}-dev python3-pip
RUN python3.11 --version

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY jwo_cv ./jwo_cv
RUN python3.11 -m jwo_cv