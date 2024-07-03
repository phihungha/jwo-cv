from __future__ import annotations

import logging
import multiprocessing as mp
from collections import Counter
from multiprocessing import connection as mpc
from typing import Sequence

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from ultralytics.utils import plotting

from jwo_cv import action_recognizer as ar
from jwo_cv import item_detector as id
from jwo_cv import shop_event
from jwo_cv.utils import Config

logger = logging.getLogger(__name__)

TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
TEXT_THICKNESS = cv2.FONT_HERSHEY_COMPLEX
TEXT_LINE_STYLE = cv2.LINE_AA

ITEM_ANNO_BOX_COLOR = (0, 0, 255)
ANNO_TEXT_COLOR = (255, 255, 255)
HAND_ANNO_BOX_COLOR = (255, 0, 0)
ANNO_LINE_WEIGHT = 2

ACTION_TEXT_ORIGIN = (10, 10)
ACTION_TEXT_COLOR = (0, 255, 0)
ACTION_TEXT_SCALE = 2


def annotate_debug_info(
    image: MatLike,
    hands: Sequence[id.Detection],
    items: Sequence[id.Detection],
    action: ar.Action | None,
) -> np.ndarray:
    """Annotate provided image with debug info.

    Args:
        image (MatLike): Image
        hands (Sequence[id.Detection]): Detected hands
        items (Sequence[id.Detection]): Detected items

    Returns:
        np.ndarray: Annotated image
    """

    annotator = plotting.Annotator(image)

    for hand in hands:
        logger.debug(hand)
        annotator.box_label(
            hand.box.to_xyxy_arr(),
            f"Hand ({round(hand.confidence, 3):.1%})",
            HAND_ANNO_BOX_COLOR,
            ANNO_TEXT_COLOR,
        )

    for item in items:
        logger.debug(item)
        annotator.box_label(
            item.box.to_xyxy_arr(),
            f"{item.class_name} ({round(item.confidence, 3):.1%})",
            ITEM_ANNO_BOX_COLOR,
            ANNO_TEXT_COLOR,
        )

    image = annotator.result()

    if action:
        text = f"Action: {action.type} ({action.confidence:.2%})"
        cv2.putText(
            image,
            text,
            org=ACTION_TEXT_ORIGIN,
            fontFace=TEXT_FONT,
            fontScale=ACTION_TEXT_SCALE,
            color=ACTION_TEXT_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotator.result()


def analyze_video(
    config: Config,
    device: str,
    frame_conn: mpc.PipeConnection,
    shop_event_queue: mp.Queue[shop_event.ShopEvent],
    use_debug_video: bool,
):
    """Analyze video for shopping events.

    Args:
        config (Config): App config
        device (str): Device to run vision ML models on
        frame_conn (mpc.PipeConnection): Video frame (numpy.NDArray) pipe connection
        to main process
        shop_event_queue (mp.Queue[shop_event.ShopEvent]): Shopping event queue
        use_debug_video (bool): Create debug video
    """

    torch.set_grad_enabled(False)
    analyzers_config = config["analyzers"]
    action_recognizer = ar.ActionRecognizer.from_config(
        analyzers_config["action"], device
    )
    item_detector = id.ItemDetector.from_config(analyzers_config)

    while True:
        frame = frame_conn.recv()
        if frame is None:
            return

        action = action_recognizer.recognize(frame)

        if use_debug_video or action:
            items, hands = item_detector.detect(frame)

        if use_debug_video:
            debug_frame = annotate_debug_info(frame, hands, items, action)
            frame_conn.send(debug_frame)

        if action and items:
            item_counts = dict(Counter(map(lambda i: i.class_id, items)))
            event = shop_event.ShopEvent(action.type, item_counts)
            logger.info(event)
            shop_event_queue.put(event)
