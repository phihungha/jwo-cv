from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from concurrent import futures
from dataclasses import dataclass

import socketio

from jwo_cv import action_detector as ad

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShoppingEvent:
    """Describes a shopping action with type (pick or return), item names and counts."""

    type: ad.ActionType
    item_counts: dict[int, int]

    def __str__(self) -> str:
        return f"{{type: {self.type}, item_counts: {self.item_counts}}}"


def create_message(event: ShoppingEvent) -> dict:
    """Create proper event message from a ShoppingEvent instance.

    Args:
        event (vision.ShoppingEvent): Shopping event instance

    Returns:
        dict: Event message
    """

    if event.type == ad.ActionType.PICK:
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


async def emit_shopping_events(
    url: str,
    namespace: str,
    shopping_event_queue: mp.Queue[ShoppingEvent],
):
    """Start emitting shopping events to server at provided URL.

    Args:
        server_url (str): URL of receiving server
        events (Iterator[vision.ShoppingEvent]): Shopping events
    """
    async_loop = asyncio.get_running_loop()

    async with socketio.AsyncSimpleClient() as sio:
        await sio.connect(url, namespace=namespace)
        logger.info(f"Connected to server {url} and start emitting events.")

        while True:
            event = await async_loop.run_in_executor(
                futures.ThreadPoolExecutor(max_workers=1), shopping_event_queue.get
            )
            message = create_message(event)
            await sio.emit("update", message)
