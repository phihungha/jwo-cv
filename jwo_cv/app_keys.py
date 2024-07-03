from __future__ import annotations

import asyncio
import multiprocessing.queues as mpq
import uuid
from concurrent import futures

import aiortc
from aiohttp import web

from jwo_cv.utils import Config

config = web.AppKey("config", Config)
device = web.AppKey("device", str)
video_peer_conns = web.AppKey(
    "video_peer_conns", dict[uuid.UUID, aiortc.RTCPeerConnection]
)
vision_process_executor = web.AppKey(
    "vision_process_executor", futures.ProcessPoolExecutor
)
shop_event_queue = web.AppKey("shop_event_queue", mpq.Queue)
shop_event_emitter = web.AppKey("shop_event_emitter", asyncio.Task[None])
