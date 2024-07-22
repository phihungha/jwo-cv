ARG PYTHON_VER=3.11.9
FROM python:${PYTHON_VER}-bookworm

WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -r requirements.txt
COPY jwo_cv/ jwo_cv/
EXPOSE 6000
CMD ["python", "-m", "jwo_cv"]