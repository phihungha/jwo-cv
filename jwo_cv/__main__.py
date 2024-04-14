import logging
import logging.config
import cv2
import toml
from ultralytics.utils import plotting

from jwo_cv import info
from jwo_cv.item_classifier import ItemClassifier


logger = logging.Logger("jwo-cv")

window = cv2.namedWindow("Debug view")


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
    """Process video from provided video source. Use q key to stop.

    Args:
        source (cv2.VideoCapture): Video source
    """

    while True:
        received, image = source.read()
        if not received:
            logger.error("Can't receive frame!")

        items = item_classifier.predict(image)

        annotator = plotting.Annotator(image)
        for item in items:
            logger.debug(item)
            annotator.box_label(
                item.box.tensor,
                item.class_name,
                info.ITEM_ANNOTATION_BOX_COLOR,
                info.ITEM_ANNOTATION_TEXT_COLOR,
            )

        cv2.imshow("Debug view", annotator.result())

        if cv2.waitKey(1) == ord("q"):
            break


def main():
    config = toml.load("jwo_cv/config/config.toml")
    logging.config.dictConfig(config["logging"])
    video_sources_config = config["video_sources"]
    video_source = getVideoSource(video_sources_config["source_idx"])
    models_config = config["models"]
    item_classifier = ItemClassifier(models_config["item_classifier_model_path"])

    processVideo(video_source, item_classifier)

    video_source.release()


main()
