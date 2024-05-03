import logging
import logging.config

import cv2
import toml
import torch

from jwo_cv import action_detector as ad
from jwo_cv import item_detector as id
from jwo_cv import vision
from jwo_cv.utils import Size

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
    shopping_event_generator = vision.processVideo(
        video_source, action_classifier, item_detector, use_debug_video
    )
    # NOTE: Do this in the API function.
    for event in shopping_event_generator:
        logger.info(event)

    if use_debug_video:
        cv2.destroyWindow("Debug")
    video_source.release()


if __name__ == "__main__":
    main()
