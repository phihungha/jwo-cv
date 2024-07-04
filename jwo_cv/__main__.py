import asyncio
import logging
import logging.config
import multiprocessing as mp
import os
from concurrent import futures

import toml
from aiohttp import web

from jwo_cv import app_keys, shop_event, video_client_api

logger = logging.getLogger("jwo-cv")


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
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = web.Application()

    app[app_keys.config] = app_config
    app[app_keys.vision_process_executor] = futures.ProcessPoolExecutor(
        mp_context=mp.get_context("forkserver")
    )
    queue_manager = mp.Manager()
    app[app_keys.shop_event_queue] = queue_manager.Queue()

    app.cleanup_ctx.append(setup_and_cleanup)
    app.add_routes(video_client_api.routes)

    api_port = app_config["video_client_api"]["port"]
    logger.info(f"Begin listening for video client connection offer on {api_port}.")
    web.run_app(app, port=api_port)


if __name__ == "__main__":
    main()
