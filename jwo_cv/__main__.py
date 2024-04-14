import logging
import logging.config
import cv2
import toml
from ultralytics.utils import plotting

from jwo_cv import info
from jwo_cv.item_classifier import ItemClassifier


logger = logging.getLogger("jwo-cv")


def getVideoSource(source_idx: int) -> cv2.VideoCapture:
    """Get a video source.

    Args:
        source_idx (int): Source index

    Returns:
        cv2.VideoCapture: Video source
    """

    video_source = cv2.VideoCapture(source_idx)

    if not video_source.isOpened():
        logger.error("Cannot open camera")
        exit()

    return video_source


def processVideo(source: cv2.VideoCapture, item_classifier: ItemClassifier):
    """Process video from provided video source.

    Args:
        source (cv2.VideoCapture): Video source
    """

    while True:
        received, image = source.read()
        if not received:
            logger.error("Can't receive frame!")

        items = item_classifier.predict(image)

        for item in items:
            logger.debug(item)


def processVideoWithDebug(source: cv2.VideoCapture, item_classifier: ItemClassifier):
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

        items = item_classifier.predict(image)

        annotator = plotting.Annotator(image)
        for item in items:
            logger.error(item)
            annotator.box_label(
                item.box.tensor,
                f"{item.class_name} ({round(item.confidence, 3):.1%})",
                info.ITEM_ANNOTATION_BOX_COLOR,
                info.ITEM_ANNOTATION_TEXT_COLOR,
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
    video_source = getVideoSource(video_sources_config["source_idx"])
    models_config = config["models"]
    item_classifier = ItemClassifier(models_config["item_classifier_model_path"])

    if general_config["debug_video"]:
        processVideoWithDebug(video_source, item_classifier)
    else:
        processVideo(video_source, item_classifier)

    video_source.release()


main()
