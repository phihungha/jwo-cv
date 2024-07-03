from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum

import socketio
import socketio.exceptions

from jwo_cv.utils import AppException

logger = logging.getLogger(__name__)


class ActionType(Enum):
    PICK = 0
    RETURN = 1


@dataclass(frozen=True)
class ShopEvent:
    """Describes a shopping action with type (pick or return), item names and counts."""

    type: ActionType
    item_counts: dict[int, int]

    def __str__(self) -> str:
        return f"{{type: {self.type}, item_counts: {self.item_counts}}}"


def create_message(event: ShopEvent) -> dict:
    """Create proper event message from a ShopEvent instance.

    Args:
        event (vision.ShoppingEvent): Shopping event instance

    Returns:
        dict: Event message
    """

    if event.type == ActionType.PICK:
        item_updates = [
            {"productId": name, "quantity": count}
            for name, count in event.item_counts.items()
        ]
    else:
        item_updates = [
            {"productId": name, "quantity": -count}
            for name, count in event.item_counts.items()
        ]

    return {"items": item_updates}


async def begin_emit_shop_events(
    url: str,
    namespace: str,
    event_queue: mp.Queue[ShopEvent],
) -> None:
    """Start emitting shopping events from a multiprocessing queue
    to server at provided URL.

    Args:
        url (str): Server URL
        namespace (str): Namespace
        event_queue (mp.Queue[ShopEvent]): Shopping event queue
    """

    async with socketio.AsyncSimpleClient() as sio:
        try:
            await sio.connect(url, namespace=namespace)
        except socketio.exceptions.ConnectionError as err:
            raise AppException(
                f"Failed to connect to server {url} on {namespace} to emit shop events."
            ) from err

        logger.info(
            f"Connected to server {url} on {namespace} and start emitting events."
        )

        async_loop = asyncio.get_event_loop()
        while True:
            event = await async_loop.run_in_executor(None, event_queue.get)
            message = create_message(event)
            await sio.emit("update", message)
