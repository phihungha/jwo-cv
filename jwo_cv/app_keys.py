from __future__ import annotations

import asyncio
import queue
from concurrent import futures

from aiohttp import web

from jwo_cv.utils import Config

config = web.AppKey("config", Config)
vision_process_executor = web.AppKey(
    "vision_process_executor", futures.ProcessPoolExecutor
)
shop_event_queue = web.AppKey("shop_event_queue", queue.Queue)
shop_event_emit_task = web.AppKey("shop_event_emit_task", asyncio.Task[None])
