import logging
import logging.config

import cv2
import toml
import torch

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
if video_config["source_video_path"] is not None:
    video_source = vision.getFileVideoSource(
        video_config["source_video_path"], image_size
    )
else:
    video_source = vision.getCameraVideoSource(video_config["source_idx"], image_size)

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

for event in shopping_event_generator:
    logger.info(event)

if use_debug_video:
    cv2.destroyWindow("Debug")
video_source.release()
