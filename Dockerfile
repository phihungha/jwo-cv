ARG PYTHON_VER=3.11.9
FROM python:${PYTHON_VER}-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY jwo_cv ./jwo_cv
RUN python -m jwo_cv