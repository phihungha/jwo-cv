import logging
import logging.config
import os

import cv2
import toml
import torch

from jwo_cv import action_detector as ad
from jwo_cv import api, vision
from jwo_cv import item_detector as id

if __name__ != "__main__":
    exit(0)

torch.set_grad_enabled(False)

logging.basicConfig()
logger = logging.getLogger("jwo-cv")

app_config_path = os.getenv("JWO_CV_CONFIG_PATH") or "jwo_cv/config/config.toml"
config = toml.load(app_config_path)
general_config = config["general"]

if general_config["debug_log"]:
    logging.root.setLevel(logging.DEBUG)
else:
    logging.root.setLevel(logging.INFO)


def getDevice() -> str:
    """Get device to run models on."""

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


device = getDevice()
logger.info("Use %s", device)

detectors_config = config["detectors"]
action_classifier = ad.ActionClassifier.from_config(detectors_config["action"], device)
item_detector = id.ItemDetector.from_config(detectors_config)

video_source = vision.getVideoSource(config["video_source"])

use_debug_video: bool = general_config["debug_video"]
if use_debug_video:
    cv2.namedWindow("Debug")

shopping_event_generator = vision.processVideo(
    video_source, action_classifier, item_detector, use_debug_video
)

emit_events_config = config["emit_events"]
if emit_events_config["emit"]:
    server_url = emit_events_config["server_url"]
    api.start_emitting_events(server_url, shopping_event_generator)
else:
    for event in shopping_event_generator:
        pass

video_source.release()
if use_debug_video:
    cv2.destroyWindow("Debug")
