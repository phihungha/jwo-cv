import logging
import logging.config
from dataclasses import dataclass
from typing import Iterator, Sequence

import cv2
import toml
import torch
from cv2.typing import MatLike
from ultralytics.utils import plotting

from jwo_cv import action_detector as ad
from jwo_cv import info
from jwo_cv import item_detector as id
from jwo_cv.utils import AppException, Size

APP_CONFIG_PATH = "jwo_cv/config/config.toml"

torch.set_grad_enabled(False)

logging.basicConfig()
logger = logging.getLogger("jwo-cv")


@dataclass
class ShoppingEvent:
    """Describes a shopping action with type (pick or return) and item names."""

    type: ad.ActionType
    item_names: list[str]

    def __str__(self) -> str:
        return f"{{type: {self.type}, item_names: {self.item_names}}}"


def getDevice() -> str:
    """Get device to run models on."""

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def getVideoSource(source_idx: int, image_size: Size) -> cv2.VideoCapture:
    """Get a video source from a source index.

    Args:
        source_idx (int): Source index
        image_size (Size): Image size

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


def showDebugInfo(
    image: MatLike,
    hands: Sequence[id.Detection],
    items: Sequence[id.Detection],
) -> None:
    """Annotate and show image on debug window.

    Args:
        image (MatLike): Image
        hands (Sequence[id.Detection]): Detected hands
        items (Sequence[id.Detection]): Detected items
    """

    annotator = plotting.Annotator(image)

    for hand in hands:
        logger.debug(hand)
        annotator.box_label(
            hand.box.to_xyxy_arr(),
            f"Hand ({round(hand.confidence, 3):.1%})",
            info.HAND_ANNOTATION_BOX_COLOR,
            info.ANNOTATION_TEXT_COLOR,
        )

    for item in items:
        logger.debug(item)
        annotator.box_label(
            item.box.to_xyxy_arr(),
            f"{item.class_name} ({round(item.confidence, 3):.1%})",
            info.ITEM_ANNOTATION_BOX_COLOR,
            info.ANNOTATION_TEXT_COLOR,
        )

    cv2.imshow("Debug", annotator.result())


def processVideo(
    source: cv2.VideoCapture,
    action_detector: ad.ActionClassifier,
    item_detector: id.ItemDetector,
    use_debug_video: bool,
) -> Iterator[ShoppingEvent]:
    """Process video and yield detected shopping events.

    Args:
        source (cv2.VideoCapture): Video source
        action_classifier (ac.ActionClassifier): Action detector
        item_detector (detectors.ItemDetector): Item detector
        use_debug_video (bool): Debug with video

    Raises:
        AppException: Failed to receive frame

    Yields:
        ShoppingEvent: Shopping event
    """

    while True:
        received, image = source.read()
        if not received:
            raise AppException("Failed to receive frame!")

        action = action_detector.detect(image)

        if use_debug_video or action:
            items, hands = item_detector.detect(image)
            item_names = list(map(lambda i: i.class_name, items))

        if use_debug_video:
            showDebugInfo(image, hands, items)
            if cv2.waitKey(1) == ord("q"):
                return

        if action:
            yield ShoppingEvent(action.type, item_names)


def main() -> None:
    config = toml.load(APP_CONFIG_PATH)
    general_config = config["general"]

    if general_config["debug_log"]:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    video_config = config["video_sources"]
    image_size = Size.from_wh_arr(video_config["size"])
    video_source = getVideoSource(video_config["source_idx"], image_size)

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
    shopping_event_generator = processVideo(
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
