from __future__ import annotations

import asyncio
import logging
import queue
from dataclasses import dataclass
from enum import Enum

import kafka

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
    kafka_producer: kafka.KafkaProducer,
    event_queue: queue.Queue[ShopEvent],
) -> None:
    """Start emitting shopping events from a multiprocessing queue
    to Kafka.

    Args:
        kafka_producer (kafka.KafkaProducer): Kafka producer
        event_queue (mp.Queue[ShopEvent]): Shopping event queue
    """

    async_loop = asyncio.get_event_loop()
    while True:
        try:
            event = await async_loop.run_in_executor(None, event_queue.get)
        except asyncio.CancelledError:
            return
        message = create_message(event)
        kafka_producer.send("cart-updates", message)
