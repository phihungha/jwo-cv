from __future__ import annotations

import asyncio
import multiprocessing.queues as mpq
from concurrent import futures

import aiortc
from aiohttp import web

from jwo_cv.utils import Config

ITEM_ANNOTATION_BOX_COLOR = (0, 0, 255)
ANNOTATION_TEXT_COLOR = (255, 255, 255)
HAND_ANNOTATION_BOX_COLOR = (255, 0, 0)
ANNOTATION_LINE_WEIGHT = 2


config_key = web.AppKey("config", Config)
device_key = web.AppKey("device", str)
video_peer_conns_key = web.AppKey("video_peer_conns", set[aiortc.RTCPeerConnection])
video_process_executor_key = web.AppKey(
    "video_process_executor", futures.ProcessPoolExecutor
)
shopping_event_queue_key = web.AppKey("shopping_event_queue", mpq.Queue)
shopping_event_emitter_key = web.AppKey("shopping_event_emitter", asyncio.Task[None])
