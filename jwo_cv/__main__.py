import asyncio
import logging
import logging.config
import multiprocessing as mp
import os
from concurrent import futures

import toml
import torch
from aiohttp import web

from jwo_cv import app_keys, shop_event, video_client_api

logger = logging.getLogger("jwo-cv")


def get_device() -> str:
    """Get device to run vision ML models on."""

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


async def setup_and_cleanup(app: web.Application):
    shop_event_config = app[app_keys.config]["shop_event"]
    if shop_event_config["emit"]:
        url = shop_event_config["url"]
        namespace = shop_event_config["namespace"]
        shop_event_queue = app[app_keys.shop_event_queue]

        app[app_keys.shop_event_emit_task] = asyncio.create_task(
            shop_event.begin_emit_shop_events(url, namespace, shop_event_queue)
        )

    yield

    logger.info("Shutting down...")

    app[app_keys.vision_process_executor].shutdown()
    logger.debug("Shut down vision worker processes.")

    video_peer_conns = app[app_keys.video_client_conns]
    await asyncio.gather(conn.close() for conn in video_peer_conns.values())
    video_peer_conns.clear()
    logger.debug("Closed all video client connections.")

    if shop_event_config["emit"]:
        shop_event_emit_task = app[app_keys.shop_event_emit_task]
        shop_event_emit_task.cancel()
        await shop_event_emit_task
        logger.debug("Stopped emitting shopping events.")


def main():
    app_config_path = os.getenv("JWO_CV_CONFIG_PATH") or "jwo_cv/config.dev.toml"
    app_config = toml.load(app_config_path)
    general_config = app_config["general"]

    if general_config["debug_log"]:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    device = get_device()
    logger.info("Use %s to run vision ML models.", device)

    app = web.Application()

    app[app_keys.config] = app_config
    app[app_keys.device] = device
    app[app_keys.shop_event_queue] = mp.Queue()
    app[app_keys.video_client_conns] = dict()
    app[app_keys.vision_process_executor] = futures.ProcessPoolExecutor(
        mp_context=mp.get_context("forkserver")
    )
    app.cleanup_ctx.append(setup_and_cleanup)
    app.add_routes(video_client_api.routes)

    api_port = app_config["video_client_api"]["port"]
    logger.info(f"Begin listening for video client connection offer on {api_port}.")
    web.run_app(app, port=api_port)


if __name__ == "__main__":
    main()
