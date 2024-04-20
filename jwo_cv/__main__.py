import logging
import logging.config
import cv2
import toml
from jwo_cv.utils import Size

from jwo_cv import info
from jwo_cv.item_detector import HandDetector, ItemDetector


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

        hands = hand_detector.detect()
        hand_boxes = list(map(lambda i: i.box, hands))

        items = item_detector.detect(image, hand_boxes)

        for item in items:
            logger.debug(item)


def processVideoWithDebug(
    source: cv2.VideoCapture,
    item_detector: ItemDetector,
    hand_detector: HandDetector,
    image_size: Size,
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

        img_blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, image_size.to_wh_seq(), swapRB=True
        )

        hands = hand_detector.detect()
        hand_boxes = list(map(lambda i: i.box, hands))
        for box in hand_boxes:
            cv2.rectangle(
                image,
                box.top_left.to_xy_list(),
                box.bot_right.to_xy_list(),
                info.HAND_ANNOTATION_BOX_COLOR,
                info.ANNOTATION_LINE_WEIGHT,
            )

        items = item_detector.detect(img_blob, hand_boxes)
        for item in items:
            logger.debug(item)
            cv2.rectangle(
                image,
                item.box.top_left.to_xy_list(),
                item.box.bot_right.to_xy_list(),
                info.ITEM_ANNOTATION_BOX_COLOR,
                info.ANNOTATION_LINE_WEIGHT,
            )
            cv2.putText(
                image,
                f"{item.class_name} ({item.confidence:.1%})",
                item.box.bot_right.to_xy_list(),
                cv2.FONT_HERSHEY_PLAIN,
                0.5,
                info.ANNOTATION_TEXT_COLOR,
            )

        cv2.imshow("Debug view", image)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyWindow("Debug view")


def main():
    config = toml.load("jwo_cv/config/config.toml")
    general_config = config["general"]

    logging.config.dictConfig(config["logging"])

    video_config = config["video_sources"]
    image_size = Size.from_wh_seq(video_config["size"])
    video_source = getVideoSource(video_config["source_idx"], image_size)

    detectors_config = config["detectors"]
    item_detector = ItemDetector.from_config(
        detectors_config["item_detector"], image_size
    )
    hand_detector = HandDetector()

    if general_config["debug_video"]:
        processVideoWithDebug(video_source, item_detector, hand_detector, image_size)
    else:
        processVideo(video_source, item_detector, hand_detector)

    video_source.release()


main()
