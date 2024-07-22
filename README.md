# Computer Vision component of JWO Shopping System

Recognize shopping actions and product items from real-time video stream.

## Technologies

- YOLOv8 via Ultralytics
- Movinet via PyTorch
- OpenCV
- Apache Kafka
- aiohttp
- WebRTC via aiortc

## Members

- Le Quang Trung
- Ha Phi Hung

## How to setup

1. Install Poetry ([Link](https://python-poetry.org/docs/#installing-with-pipx))
2. Install Pyenv ([Link](https://github.com/pyenv/pyenv?tab=readme-ov-file#getting-pyenv))
3. Run `pyenv install 3.11.9`
4. Clone this repository
5. Run `poetry install` in repository folder

## How to run

1. Run `poetry shell` in repository folder to enter virtual environment
2. Run `python -m jwo_cv`

Note: Set `emit` to `false` in `jwo_cv/config/config.dev.toml` to disable emitting shopping events