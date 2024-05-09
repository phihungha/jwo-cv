import logging
import logging.config
import cv2
import toml
import torch
from random import randint

from jwo_cv import action_detector as ad
from jwo_cv import item_detector as id
from jwo_cv import vision
from jwo_cv.utils import Size

from flask import Flask, Response

import time
import json


APP_CONFIG_PATH = "jwo_cv/config/config.toml"
shopping_event_generator = {}
app = Flask(__name__) 
def event_stream():
    while True:
        time.sleep(0.005)
        for event in shopping_event_generator:
            msg = {"type": event.type, "item_names": event.item_names}
            data = json.dumps(msg)
        yield f"event:{event}\ndata:{data}\n\n"
@app.route("/")
def stream():
    return Response(event_stream(), mimetype="text/event-stream")  
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


def main() -> None:
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

    action_classifier = ad.ActionClassifier.from_config(
        detectors_config["action"], device
    )
    item_detector = id.ItemDetector.from_config(detectors_config)

    use_debug_video: bool = general_config["debug_video"]

    if use_debug_video:
        cv2.namedWindow("Debug")

    # NOTE: Find a way to pass this generator into the API function.
    global shopping_event_generator
    shopping_event_generator = vision.processVideo(
        video_source, action_classifier, item_detector, use_debug_video
    )
    app.run(debug=True)
    # NOTE: Do this in the API function.

    if use_debug_video:
        cv2.destroyWindow("Debug")
    video_source.release()

if __name__ == "__main__": 
    main()
    
