import logging
from dataclasses import dataclass
from typing import Iterator, Sequence

import cv2
from cv2.typing import MatLike
from ultralytics.utils import plotting

from jwo_cv import action_detector as ad
from jwo_cv import info
from jwo_cv import item_detector as id
from jwo_cv.utils import AppException, Size

logger = logging.getLogger(__name__)


@dataclass
class ShoppingEvent:
    """Describes a shopping action with type (pick or return) and item names."""

    type: ad.ActionType
    item_names: list[str]

    def __str__(self) -> str:
        return f"{{type: {self.type}, item_names: {self.item_names}}}"


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
