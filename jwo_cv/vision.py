from __future__ import annotations

import logging
import multiprocessing as mp
from collections import Counter
from typing import Sequence

import cv2
import torch
from cv2.typing import MatLike
from numpy import typing as np_types
from ultralytics.utils import plotting

from jwo_cv import action_detector as ad
from jwo_cv import info
from jwo_cv import item_detector as id
from jwo_cv import shopping_event as se
from jwo_cv.utils import Config

logger = logging.getLogger(__name__)


def show_debug_info(
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


def process_video(
    client_id: str | None,
    config: Config,
    device: str,
    video_frame_queue: mp.Queue[np_types.NDArray],
    shopping_event_queue: mp.Queue[se.ShoppingEvent],
):
    torch.set_grad_enabled(False)

    detectors_config = config["detectors"]
    action_detector = ad.ActionClassifier.from_config(
        detectors_config["action"], device
    )
    item_detector = id.ItemDetector.from_config(detectors_config)

    use_debug_video: bool = False
    window_name = f"Video from client {client_id}"
    if use_debug_video:
        cv2.namedWindow(window_name)

    while True:
        image = video_frame_queue.get()

        action = action_detector.detect(image)

        if use_debug_video or action:
            items, hands = item_detector.detect(image)

        if use_debug_video:
            show_debug_info(image, hands, items)
            if cv2.waitKey(1) == ord("q"):
                break

        if action and items:
            item_counts = dict(Counter(map(lambda i: i.class_id, items)))
            event = se.ShoppingEvent(action.type, item_counts)
            logger.info(event)
            shopping_event_queue.put(event)

    if use_debug_video:
        cv2.destroyWindow(window_name)
