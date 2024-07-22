import asyncio
import json
import logging
import logging.config
import multiprocessing as mp
import os
from concurrent import futures

import kafka
import kafka.errors
import movinets
import movinets.config
import toml
from aiohttp import web

from jwo_cv import (
    action_recognizer as ar,
)
from jwo_cv import (
    app_keys,
    shop_event,
    utils,
    video_client_api,
)


def setup_logging():
    log_handlers = utils.get_log_handlers()

    if os.getenv(utils.DEBUG_ENV_VAR) == "1":
        logging.basicConfig(level=logging.DEBUG, handlers=log_handlers)
    else:
        logging.basicConfig(level=logging.INFO, handlers=log_handlers)


setup_logging()
logger = logging.getLogger("jwo_cv")


async def setup_and_cleanup(app: web.Application):
    shop_event_config = app[app_keys.config]["shop_event"]
    if shop_event_config["emit"]:
        kafka_broker_server = shop_event_config["broker"]
        try:
            kafka_producer = kafka.KafkaProducer(
                bootstrap_servers=kafka_broker_server,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info("Connected to Kafka broker at %s", kafka_broker_server)
        except kafka.errors.NoBrokersAvailable as err:
            raise utils.AppException("Kafka broker not available.") from err

        shop_event_queue = app[app_keys.shop_event_queue]
        app[app_keys.shop_event_emit_task] = asyncio.create_task(
            shop_event.begin_emit_shop_events(kafka_producer, shop_event_queue)
        )

    yield

    logger.info("Shutting down...")

    app[app_keys.vision_process_executor].shutdown()
    logger.debug("Shut down vision worker processes.")

    if shop_event_config["emit"]:
        shop_event_emit_task = app[app_keys.shop_event_emit_task]
        shop_event_emit_task.cancel()
        await shop_event_emit_task
        kafka_producer.flush()
        logger.debug("Stopped emitting shopping events.")


def main():
    app_config_path = os.getenv("JWO_CV_CONFIG_PATH") or "config.toml"
    app_config = toml.load(app_config_path)

    # Download pre-trained weights if not exist
    movinets.MoViNet(ar.MODEL_CONFIG, causal=True, pretrained=True)

    app = web.Application()
    app[app_keys.config] = app_config
    app[app_keys.vision_process_executor] = futures.ProcessPoolExecutor(
        mp_context=mp.get_context("forkserver")
    )

    if app_config["shop_event"]["emit"]:
        queue_manager = mp.Manager()
        app[app_keys.shop_event_queue] = queue_manager.Queue()
    else:
        app[app_keys.shop_event_queue] = None

    app.cleanup_ctx.append(setup_and_cleanup)
    app.add_routes(video_client_api.routes)

    api_port = app_config["video_client_api"]["port"]
    logger.info(f"Begin listening for video client connection offer on {api_port}.")
    web.run_app(app, port=api_port)


if __name__ == "__main__":
    try:
        main()
    except utils.AppException as err:
        logger.error(err)
    except Exception as err:
        logger.exception(err)
