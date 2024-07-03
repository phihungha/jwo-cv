import logging
import multiprocessing as mp

import aiortc
from aiohttp import web
from aiortc.contrib import media

from jwo_cv import info, vision
from jwo_cv.utils import AppException

logger = logging.Logger(__name__)


routes = web.RouteTableDef()


@routes.post("/offer")
async def offer(req: web.Request):
    req_body = await req.json()
    offer = aiortc.RTCSessionDescription(sdp=req_body["sdp"], type=req_body["type"])

    peer_conn = aiortc.RTCPeerConnection()
    peer_conns = req.app[info.video_peer_conns_key]
    peer_conns.add(peer_conn)

    @peer_conn.on("connectionstatechange")
    async def on_conn_state_change():
        logger.info("Video client's connection state is ", peer_conn.connectionState)
        if peer_conn.connectionState == "failed":
            await peer_conn.close()
            peer_conns.discard(peer_conn)

    @peer_conn.on("track")
    async def on_track(track: aiortc.MediaStreamTrack):
        if track.kind == "video":
            relay = media.MediaRelay()
            await process_video_track(relay.subscribe(track), req)

    await peer_conn.setRemoteDescription(offer)

    answer = await peer_conn.createAnswer()
    if answer is None:
        raise AppException(
            "Failed to create answer to video client's connection offer."
        )
    await peer_conn.setLocalDescription(answer)

    resp_body = {
        "sdp": peer_conn.localDescription.sdp,
        "type": peer_conn.localDescription.type,
    }
    return web.json_response(resp_body)


async def process_video_track(track: aiortc.MediaStreamTrack, req: web.Request):
    print("process")
    video_frame = await track.recv()
    app = req.app
    client_id = req.remote
    process_executor = app[info.video_process_executor_key]
    shopping_event_queue = app[info.shopping_event_queue_key]
    video_frame_queue = mp.Queue()

    process_executor.submit(
        vision.process_video,
        client_id,
        app[info.config_key],
        app[info.device_key],
        video_frame_queue,
        shopping_event_queue,
    )

    while True:
        decoded_video_frame = video_frame.to_ndarray(format="bgr24")  # type: ignore
        video_frame_queue.put(decoded_video_frame)
        video_frame = await track.recv()
