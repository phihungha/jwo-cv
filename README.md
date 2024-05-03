# Computer Vision component of JWO Shopping System

Detect shopping actions and product items from real-time video then stream them via a real-time API.

## Technologies

- YOLOv8 via Ultralytics
- Movinet via PyTorch
- OpenCV
- Flask API + WebSocket

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

Note: Set `debug_video` to false in `jwo_cv/config/config.toml` to disable video display