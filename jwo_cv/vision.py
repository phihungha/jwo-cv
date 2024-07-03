import logging
import multiprocessing as mp
from collections import Counter
from multiprocessing import connection as mpc
from typing import Sequence

import numpy as np
import torch
from cv2.typing import MatLike
from ultralytics.utils import plotting

from jwo_cv import action_detector as ad
from jwo_cv import item_detector as id
from jwo_cv import shop_event
from jwo_cv.utils import Config

logger = logging.getLogger(__name__)

ITEM_ANNO_BOX_COLOR = (0, 0, 255)
ANNO_TEXT_COLOR = (255, 255, 255)
HAND_ANNO_BOX_COLOR = (255, 0, 0)
ANNO_LINE_WEIGHT = 2


def annotate_debug_info(
    image: MatLike,
    hands: Sequence[id.Detection],
    items: Sequence[id.Detection],
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

    return annotator.result()


def analyze_video(
    config: Config,
    device: str,
    frame_conn: mpc.Connection,
    shop_event_queue: mp.Queue[shop_event.ShopEvent],
    use_debug_video: bool,
):
    """Analyze video for shopping events.

    Args:
        config (Config): App config
        device (str): Device to run vision ML models on
        frame_conn (mpc.Connection): Video frame pipe connection to main process
        shop_event_queue (mp.Queue[shop_event.ShopEvent]): Shopping event queue
        use_debug_video (bool): Create debug video
    """

    torch.set_grad_enabled(False)
    detectors_config = config["detectors"]
    action_detector = ad.ActionClassifier.from_config(
        detectors_config["action"], device
    )
    item_detector = id.ItemDetector.from_config(detectors_config)

    while True:
        frame = frame_conn.recv()
        if frame is None:
            return

        action = action_detector.detect(frame)

        if use_debug_video or action:
            items, hands = item_detector.detect(frame)

        if use_debug_video:
            debug_frame = annotate_debug_info(frame, hands, items)
            frame_conn.send(debug_frame)

        if action and items:
            item_counts = dict(Counter(map(lambda i: i.class_id, items)))
            event = shop_event.ShopEvent(action.type, item_counts)
            logger.info(event)
            shop_event_queue.put(event)
