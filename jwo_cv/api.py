import logging
from collections.abc import Iterator

import socketio

from jwo_cv import action_detector as ad
from jwo_cv import vision

logger = logging.getLogger(__name__)


def create_message(event: vision.ShoppingEvent) -> list[dict]:
    """Create proper event message from a ShoppingEvent instance.

    Args:
        event (vision.ShoppingEvent): Shopping event instance

    Returns:
        list[dict]: Event message
    """

    if event.type == ad.ActionType.PICK:
        message = [
            {"productId": name, "quantity": count}
            for name, count in event.item_counts.items()
        ]
    else:
        message = [
            {"productId": name, "quantity": -count}
            for name, count in event.item_counts.items()
        ]

    return message


def start_emitting_events(server_url: str, events: Iterator[vision.ShoppingEvent]):
    """Start emitting shopping events to server at provided URL.

    Args:
        server_url (str): URL of receiving server
        events (Iterator[vision.ShoppingEvent]): Shopping events
    """

    with socketio.SimpleClient() as sio:
        sio.connect(server_url)
        logger.info(f"Connected to server {server_url} and start emitting events.")

        for event in events:
            message = create_message(event)
            sio.emit("update", message)
