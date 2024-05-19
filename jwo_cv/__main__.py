import json
import logging
import logging.config
import time

import cv2
import toml
import torch
from flask import Flask
from flask_socketio import SocketIO, emit

from jwo_cv import action_detector as ad
from jwo_cv import item_detector as id
from jwo_cv import vision
from jwo_cv.utils import Size

if __name__ != "__main__":
    exit(0)

APP_CONFIG_PATH = "jwo_cv/config/config.toml"

torch.set_grad_enabled(False)

logging.basicConfig()
logger = logging.getLogger("jwo-cv")


def getDevice() -> str:
    """Get device to run models on."""

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


config = toml.load(APP_CONFIG_PATH)
general_config = config["general"]

if general_config["debug_log"]:
    logging.root.setLevel(logging.DEBUG)
else:
    logging.root.setLevel(logging.INFO)


video_config = config["video_source"]
image_size = Size.from_wh_arr(video_config["size"])
video_source = vision.getVideoSource(video_config["source_idx"], image_size)

detectors_config = config["detectors"]
device = getDevice()
logger.info("Use %s", device)

action_classifier = ad.ActionClassifier.from_config(detectors_config["action"], device)
item_detector = id.ItemDetector.from_config(detectors_config)

use_debug_video: bool = general_config["debug_video"]

if use_debug_video:
    cv2.namedWindow("Debug")

shopping_event_generator = vision.processVideo(
    video_source, action_classifier, item_detector, use_debug_video
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
@socketio.on("connect_video")
def event_stream():
    for event in shopping_event_generator:
        logger.info(event)
        msg = {
            "time": time.time(),
            "type": str(event.type),
            "item_names": event.item_names,
        }
        emit("Video", json.dumps(msg), broadcast=True)
        socketio.sleep(1)


if general_config["api"]:
    socketio.run(app)
else:
    for event in shopping_event_generator:
        logger.info(event)


if use_debug_video:
    cv2.destroyWindow("Debug")
video_source.release()
