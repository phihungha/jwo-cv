from __future__ import annotations

import queue
from collections import Counter
from typing import Sequence

import cv2
import numpy as np
from cv2 import typing as cv2_t
from ultralytics.utils import plotting

from jwo_cv import action_recognizer as ar
from jwo_cv import item_detector as id
from jwo_cv import shop_event, utils
from jwo_cv.utils import AppException, Config

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_LINE_STYLE = cv2.LINE_AA

ITEM_ANNO_BOX_COLOR = (0, 0, 255)
ANNO_TEXT_COLOR = (255, 255, 255)
HAND_ANNO_BOX_COLOR = (255, 0, 0)
ANNO_LINE_WEIGHT = 2

ACTION_TEXT_ORIGIN = (50, 50)
ACTION_TEXT_COLOR = (0, 255, 0)
ACTION_TEXT_SCALE = 0.6

logger = utils.get_multiprocess_logger()


def annotate_debug_info(
    frame: cv2_t.MatLike,
    hands: Sequence[id.Detection],
    items: Sequence[id.Detection],
    actions: Sequence[ar.Action],
) -> np.ndarray:
    """Annotate provided image with debug info.

    Args:
        frame (cv2_t.MatLike): Image
        hands (Sequence[id.Detection]): Hand detections
        items (Sequence[id.Detection]): Item detections

    Returns:
        np.ndarray: Annotated image
    """

    annotator = plotting.Annotator(frame)

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

    frame = annotator.result()

    for i, action in enumerate(actions):
        text = f"{action.type.name}: {action.confidence:.2%}"
        cv2.putText(
            frame,
            text,
            org=(ACTION_TEXT_ORIGIN[0], ACTION_TEXT_ORIGIN[1] + i * 30),
            fontFace=TEXT_FONT,
            fontScale=ACTION_TEXT_SCALE,
            color=ACTION_TEXT_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotator.result()


class VisionAnalyzer:
    """Analyzes video frames for shopping actions and related items
    then sends it to provided event queue.
    """

    def __init__(
        self,
        action_recognizer: ar.ActionRecognizer,
        item_detector: id.ItemDetector,
        event_queue: queue.Queue[shop_event.ShopEvent] | None,
    ):
        """Analyzes video frames for shopping actions and related items
        then sends it to provided event queue.

        Args:
            action_recognizer (ar.ActionRecognizer): Action recognizer
            item_detector (id.ItemDetector): Item detector
            event_queue (queue.Queue[shop_event.ShopEvent] | None): Event queue
        """

        self.action_recognizer = action_recognizer
        self.item_detector = item_detector
        self.event_queue = event_queue
        self.just_detected = False

    @classmethod
    def from_config(
        cls,
        config: Config,
        event_queue: queue.Queue[shop_event.ShopEvent] | None,
    ):
        action_recognizer = ar.ActionRecognizer.from_config(config["action"])
        item_detector = id.ItemDetector.from_config(config)

        return VisionAnalyzer(action_recognizer, item_detector, event_queue)

    def analyze_video_frame(
        self, frame: cv2_t.MatLike, debug=False
    ) -> np.ndarray | None:
        """Analyze provided video frame for a shopping event
        then send it to event queue.

        Args:
            frame (cv2_t.MatLike): Video frame
            debug (bool): Return frame with debug info

        Returns:
            np.ndarray | None: frame with debug info
        """

        items, hands = self.item_detector.detect(frame)

        if items:
            actions, recognized_action = self.action_recognizer.recognize(frame)
        else:
            self.action_recognizer.model.clean_activation_buffers()
            actions, recognized_action = [], None

        if not items:
            self.just_detected = False

        if recognized_action and items and not self.just_detected:
            item_counts = dict(Counter(map(lambda i: i.class_id, items)))
            event = shop_event.ShopEvent(recognized_action.type, item_counts)
            self.just_detected = True
            logger.info(event)

            if self.event_queue is not None:
                try:
                    self.event_queue.put(event, timeout=10)
                except queue.Full:
                    raise AppException("Vision event queue is full.")

        if debug:
            return annotate_debug_info(frame, hands, items, actions)
