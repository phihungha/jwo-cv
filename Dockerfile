FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY jwo_cv ./jwo_cv
RUN python -m jwo_cv