from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import uuid
from multiprocessing import connection as mpc

import aiortc
import numpy as np
from aiohttp import web
from aiortc.contrib import media

from jwo_cv import app_keys, vision
from jwo_cv.utils import AppException

logger = logging.Logger(__name__)
media_relay = media.MediaRelay()
routes = web.RouteTableDef()


@routes.post("/offer")
async def offer(req: web.Request):
    """Receive offer to establish WebRTC peer connection for video streaming."""

    req_body = await req.json()
    if "sdp" not in req_body or "type" not in req_body:
        raise web.HTTPBadRequest(text="Invalid video connection offer.")

    use_debug_video: bool = req_body.get("use_debug_video", False)

    offer = aiortc.RTCSessionDescription(sdp=req_body["sdp"], type=req_body["type"])

    peer_conn = aiortc.RTCPeerConnection()
    peer_conn_id = uuid.uuid4()

    peer_conns = req.app[app_keys.video_client_conns]

    @peer_conn.on("connectionstatechange")
    async def on_conn_state_change():
        logger.info(
            "Video client %s connection state is %s",
            peer_conn_id,
            peer_conn.connectionState,
        )

        if peer_conn.connectionState == "failed":
            await peer_conn.close()
            del peer_conns[peer_conn_id]

        if peer_conn.connectionState == "closed":
            try:
                del peer_conns[peer_conn_id]
            except KeyError:
                pass

    media_blackhole = media.MediaBlackhole()

    @peer_conn.on("track")
    def on_track(track: aiortc.MediaStreamTrack):
        if track.kind != "video":
            return

        vision_track = VideoVisionTrack.from_track(track, req.app, use_debug_video)
        if use_debug_video:
            peer_conn.addTrack(vision_track)
        else:
            media_blackhole.addTrack(vision_track)

    await peer_conn.setRemoteDescription(offer)
    await media_blackhole.start()

    answer = await peer_conn.createAnswer()
    if answer is None:
        raise AppException(
            "Failed to create answer to video client's connection offer."
        )
    await peer_conn.setLocalDescription(answer)

    peer_conns[peer_conn_id] = peer_conn

    resp_body = {
        "id": peer_conn_id,
        "sdp": peer_conn.localDescription.sdp,
        "type": peer_conn.localDescription.type,
    }
    return web.json_response(resp_body)


class VideoVisionTrack(aiortc.MediaStreamTrack):
    """A video stream track which performs computer vision work on video
    and returns video with debug info.
    """

    kind = "video"

    def __init__(
        self,
        input_track: aiortc.MediaStreamTrack,
        frame_conn: mpc.PipeConnection,
        use_debug_video: bool = False,
    ):
        """A video stream track which performs computer vision work on video
        and returns video with debug info.

        Args:
            input_track (aiortc.MediaStreamTrack): Receiving video track
            frame_conn (mpc.PipeConnection): Video frame (numpy.NDArray) pipe connection
           to analysis worker process
            use_debug_video (bool, optional): Return debug video stream.
            Defaults to False.
        """

        super().__init__()

        self.input_track = input_track
        self.frame_conn = frame_conn
        self.use_debug_video = use_debug_video

    @classmethod
    def from_track(
        cls, track: aiortc.MediaStreamTrack, app: web.Application, use_debug_video: bool
    ) -> VideoVisionTrack:
        """Create VideoAnalyzeTrack from a video track and start
        corresponding vision worker process.

        Args:
            track (aiortc.MediaStreamTrack): Input video track
            app (web.Application): Application
            use_debug_video (bool): Return debug video stream

        Returns:
            VideoAnalyzeTrack
        """

        frame_main_conn, frame_worker_conn = mp.Pipe()
        shop_event_queue = app[app_keys.shop_event_queue]

        process_executor = app[app_keys.vision_process_executor]
        process_executor.submit(
            vision.analyze_video,
            app[app_keys.config],
            app[app_keys.device],
            frame_worker_conn,
            shop_event_queue,
            use_debug_video,
        )

        track = media_relay.subscribe(track)
        return cls(track, frame_main_conn, use_debug_video)

    async def recv(self):
        video_frame = (await self.input_track.recv()).to_ndarray(format="bgr24")  # type: ignore

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.frame_conn.send, video_frame)

        if self.use_debug_video:
            return_video_frame = await loop.run_in_executor(None, self.frame_conn.recv)
        else:
            return_video_frame = np.zeros_like(video_frame)

        return return_video_frame
