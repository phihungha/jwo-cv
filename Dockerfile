ARG PYTHON_VER=3.11.9
FROM python:${PYTHON_VER}-bookworm
RUN apt update && apt install ffmpeg libsm6 libxext6

WORKDIR /app
COPY requirements-hashed.txt requirements-vcs.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-hashed.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-vcs.txt
COPY jwo_cv/ jwo_cv/
RUN python -m jwo_cv