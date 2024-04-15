import logging
import logging.config
from typing import Sequence
import cv2
import toml
from ultralytics.utils import plotting

from jwo_cv import info
from jwo_cv.item_detector import HandDetector, ItemDetector


logger = logging.getLogger("jwo-cv")


def getVideoSource(source_idx: int, resolution: Sequence[int]) -> cv2.VideoCapture:
    """Get a video source.

    Args:
        source_idx (int): Source index
        resolution (Sequence[int]): Image resolution as [width, height]

    Returns:
        cv2.VideoCapture: Video source
    """

    video_source = cv2.VideoCapture(source_idx)
    video_source.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[0])
    video_source.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[1])

    if not video_source.isOpened():
        logger.error("Cannot open camera")
        exit()

    return video_source


def processVideo(
    source: cv2.VideoCapture, item_detector: ItemDetector, hand_detector: HandDetector
):
    """Process video from provided video source.

    Args:
        source (cv2.VideoCapture): Video source
    """

    while True:
        received, image = source.read()
        if not received:
            logger.error("Can't receive frame!")

        hands = hand_detector.predict()
        hand_boxes = list(map(lambda i: i.box, hands))

        items = item_detector.detect(image, hand_boxes)

        for item in items:
            logger.debug(item)


def processVideoWithDebug(
    source: cv2.VideoCapture, item_detector: ItemDetector, hand_detector: HandDetector
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

        hands = hand_detector.predict()
        hand_boxes = list(map(lambda i: i.box, hands))
        for box in hand_boxes:
            annotator.box_label(
                box.to_tensor(),
                "Hand",
                info.HAND_ANNOTATION_BOX_COLOR,
                info.ANNOTATION_TEXT_COLOR,
            )

        items = item_detector.detect(image, hand_boxes)
        for item in items:
            logger.debug(item)
            annotator.box_label(
                item.box.to_tensor(),
                f"{item.class_name} ({round(item.confidence, 3):.1%})",
                info.ITEM_ANNOTATION_BOX_COLOR,
                info.ANNOTATION_TEXT_COLOR,
            )

        cv2.imshow("Debug view", annotator.result())

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyWindow("Debug view")


def main():
    config = toml.load("jwo_cv/config/config.toml")
    general_config = config["general"]

    logging.config.dictConfig(config["logging"])

    video_sources_config = config["video_sources"]
    video_source = getVideoSource(
        video_sources_config["source_idx"], video_sources_config["resolution"]
    )

    models_config = config["models"]
    item_detector = ItemDetector(
        models_config["item_detector_model_path"], models_config["max_hand_distance"]
    )
    hand_detector = HandDetector()

    if general_config["debug_video"]:
        processVideoWithDebug(video_source, item_detector, hand_detector)
    else:
        processVideo(video_source, item_detector, hand_detector)

    video_source.release()


main()
