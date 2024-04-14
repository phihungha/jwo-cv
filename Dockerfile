FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY jwo-cv .
RUN python -m jwo-cv