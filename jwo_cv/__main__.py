import asyncio
import logging
import logging.config
import multiprocessing as mp
import os
from concurrent import futures

import toml
import torch
from aiohttp import web

from jwo_cv import info, video_client_api
from jwo_cv import shopping_event as se

logger = logging.getLogger("jwo-cv")


def getDevice() -> str:
    """Get device to run models on."""

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


async def setup_and_cleanup(app: web.Application):
    emit_events_config = app[info.config_key]["shopping_events"]
    emit = emit_events_config["emit"]
    if emit:
        url = emit_events_config["url"]
        namespace = emit_events_config["namespace"]
        shopping_event_queue = app[info.shopping_event_queue_key]
        app[info.shopping_event_emitter_key] = asyncio.create_task(
            se.emit_shopping_events(url, namespace, shopping_event_queue)
        )

    yield

    app[info.video_process_executor_key].shutdown()

    video_peer_conns = app[info.video_peer_conns_key]
    await asyncio.gather(conn.close() for conn in video_peer_conns)
    video_peer_conns.clear()

    app[info.shopping_event_emitter_key].cancel()
    await app[info.shopping_event_emitter_key]


def main():
    app_config_path = os.getenv("JWO_CV_CONFIG_PATH") or "jwo_cv/config.dev.toml"
    app_config = toml.load(app_config_path)
    general_config = app_config["general"]

    if general_config["debug_log"]:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    device = getDevice()
    logger.info("Use %s", device)

    app = web.Application()
    app[info.config_key] = app_config
    app[info.device_key] = device
    app[info.shopping_event_queue_key] = mp.Queue()
    app[info.video_peer_conns_key] = set()
    app[info.video_process_executor_key] = futures.ProcessPoolExecutor(
        mp_context=mp.get_context("forkserver")
    )
    app.add_routes(video_client_api.routes)
    app.cleanup_ctx.append(setup_and_cleanup)

    port = app_config["video_client_api"]["port"]
    logger.info(f"Begin listening to video client connection offer on {port}")
    web.run_app(app, port=port)


if __name__ == "__main__":
    main()
