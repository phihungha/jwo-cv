import logging
import logging.config
import cv2
import toml

import torch
from ultralytics.utils import plotting

from jwo_cv import info
from jwo_cv import action_classifier as ac
from jwo_cv import detectors
from jwo_cv.utils import Size

torch.set_grad_enabled(False)

logging.basicConfig()
logger = logging.getLogger("jwo-cv")


def getVideoSource(source_idx: int, image_size: Size) -> cv2.VideoCapture:
    """Get a video source.

    Args:
        source_idx (int): Source index
        resolution (ImageSize): Video image size

    Returns:
        cv2.VideoCapture: Video source
    """

    video_source = cv2.VideoCapture(source_idx)
    video_source.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)
    video_source.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width)

    if not video_source.isOpened():
        logger.error("Cannot open camera")
        exit()

    return video_source


def processVideo(
    source: cv2.VideoCapture,
    action_classifier: ac.ActionClassifier,
    item_detector: detectors.ItemDetector,
    hand_detector: detectors.HandDetector,
):
    """Process video from provided video source.

    Args:
        source (cv2.VideoCapture): Video source
    """

    while True:
        received, image = source.read()
        if not received:
            logger.error("Can't receive frame!")

        action = action_classifier.predict(image)
        if action is None:
            continue

        if action.type == ac.ActionType.PICK:
            logger.info(f"Pick detected with confidence {action.confidence:.1%}")
        elif action.type == ac.ActionType.RETURN:
            logger.info(f"Return detected with confidence {action.confidence:.1%}")

        hands = hand_detector.detect(image)
        hand_boxes = list(map(lambda i: i.box, hands))
        items = item_detector.detect(image, hand_boxes)

        if items:
            logger.info("Items: ", list(map(lambda i: i.class_name, items)))
        else:
            logger.info("No item found")


def processVideoWithDebug(
    source: cv2.VideoCapture,
    action_classifier: ac.ActionClassifier,
    item_detector: detectors.ItemDetector,
    hand_detector: detectors.HandDetector,
):
    """Process video from provided video source with debug view.
    Use q key to stop.

    Args:
        source (cv2.VideoCapture): Video source
    """

    cv2.namedWindow("Debug view")

    while True:
        received, image = source.read()
        if not received:
            logger.error("Can't receive frame!")

        annotator = plotting.Annotator(image)

        hands = hand_detector.detect(image)
        for hand in hands:
            annotator.box_label(
                hand.box.to_xyxy_arr(),
                f"Hand ({round(hand.confidence, 3):.1%})",
                info.HAND_ANNOTATION_BOX_COLOR,
                info.ANNOTATION_TEXT_COLOR,
            )

        hand_boxes = list(map(lambda i: i.box, hands))
        items = item_detector.detect(image, hand_boxes)
        for item in items:
            logger.debug(item)
            annotator.box_label(
                item.box.to_xyxy_arr(),
                f"{item.class_name} ({round(item.confidence, 3):.1%})",
                info.ITEM_ANNOTATION_BOX_COLOR,
                info.ANNOTATION_TEXT_COLOR,
            )

        item_names = ", ".join(map(lambda i: i.class_name, items))

        action = action_classifier.predict(image)
        if action is not None and item_names:
            if action.type == ac.ActionType.PICK:
                logger.info(
                    f"Pick detected with confidence {action.confidence:.1%} on items {item_names}"
                )
            elif action.type == ac.ActionType.RETURN:
                logger.info(
                    f"Return detected with confidence {action.confidence:.1%} on items {item_names}"
                )

        cv2.imshow("Debug view", annotator.result())

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyWindow("Debug view")


def main():
    config = toml.load("jwo_cv/config/config.toml")
    general_config = config["general"]

    if general_config["debug_log"]:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    video_config = config["video_sources"]
    image_size = Size.from_wh_arr(video_config["size"])
    video_source = getVideoSource(video_config["source_idx"], image_size)

    detectors_config = config["detectors"]
    action_classifier = ac.ActionClassifier.from_config(detectors_config["action"])
    item_detector = detectors.ItemDetector.from_config(detectors_config["item"])
    hand_detector = detectors.HandDetector.from_config(detectors_config["hand"])

    if general_config["debug_video"]:
        processVideoWithDebug(
            video_source, action_classifier, item_detector, hand_detector
        )
    else:
        processVideo(video_source, action_classifier, item_detector, hand_detector)

    video_source.release()


main()
