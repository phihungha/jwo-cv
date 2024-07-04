from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import signal
import uuid
from multiprocessing import connection as mpc

import aiortc
from aiohttp import web
from aiortc.contrib import media
from av.video import frame as av_vframe

from jwo_cv import app_keys, shop_event, utils, vision
from jwo_cv.utils import AppException, Config

logger = utils.create_multiprocessing_logger()

media_relay = media.MediaRelay()
routes = web.RouteTableDef()


@routes.post("/offer")
async def offer(req: web.Request):
    """Receive offer to establish WebRTC peer connection for video streaming."""

    req_body = await req.json()
    if "sdp" not in req_body or "type" not in req_body:
        raise web.HTTPBadRequest(text="Invalid video connection offer.")

    use_debug_video: bool = req_body.get("use_debug_video", False)

    process_executor = req.app[app_keys.vision_process_executor]
    answer_main_conn, answer_worker_conn = mp.Pipe(duplex=False)

    process_executor.submit(
        start_video_conn,
        req_body["sdp"],
        req_body["type"],
        use_debug_video,
        req.app[app_keys.config],
        answer_worker_conn,
        req.app[app_keys.shop_event_queue],
    )

    resp_body = answer_main_conn.recv()
    return web.json_response(resp_body)


async def _start_video_conn(
    sdp: str,
    type: str,
    use_debug_video: bool,
    config: Config,
    answer_conn: mpc.Connection,
    shop_event_queue: queue.Queue[shop_event.ShopEvent] | None,
):
    offer = aiortc.RTCSessionDescription(sdp, type)

    peer_conn = aiortc.RTCPeerConnection()
    peer_conn_id = uuid.uuid4()

    loop = asyncio.get_event_loop()

    async def on_stop_signal():
        logger.info("Video client %s's connection is being shut down...", peer_conn_id)
        await peer_conn.close()
        loop.stop()

    for sig_name in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(
            getattr(signal, sig_name), lambda: asyncio.create_task(on_stop_signal())
        )

    @peer_conn.on("connectionstatechange")
    async def on_conn_state_change():
        logger.info(
            "Video client %s's connection state is '%s'",
            peer_conn_id,
            peer_conn.connectionState,
        )

        if peer_conn.connectionState == "failed":
            await peer_conn.close()

        if peer_conn.connectionState == "closed":
            loop.stop()

    media_blackhole = media.MediaBlackhole()

    @peer_conn.on("track")
    def on_track(track: aiortc.MediaStreamTrack):
        if track.kind != "video":
            logger.debug(
                "Video client %s has non-video track. Ignoring...", peer_conn_id
            )
            return

        vision_analyzer = vision.VisionAnalyzer.from_config(
            config["analyzers"], shop_event_queue
        )
        vision_track = VideoVisionTrack.from_video_track(
            track, vision_analyzer, use_debug_video
        )
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

    resp_body = {
        "id": str(peer_conn_id),
        "sdp": peer_conn.localDescription.sdp,
        "type": peer_conn.localDescription.type,
    }
    answer_conn.send(resp_body)


def start_video_conn(
    sdp: str,
    type: str,
    use_debug_video: bool,
    config: Config,
    answer_conn: mpc.Connection,
    shop_event_queue: queue.Queue[shop_event.ShopEvent] | None,
) -> None:
    """Start video stream WebRTC connection from provided offer
    and perform computer vision work on it.

    Args:
        sdp (str): Offer SDP
        type (str): Offer type
        use_debug_video (bool): Return debug video to client
        config (Config): App config
        answer_conn (mpc.Connection): Pipe connection to return peer connection answer
        shop_event_queue (queue.Queue[shop_event.ShopEvent] | None): Shop event queue
    """
    try:
        event_loop = asyncio.get_event_loop()
        event_loop.create_task(
            _start_video_conn(
                sdp, type, use_debug_video, config, answer_conn, shop_event_queue
            )
        )
        event_loop.run_forever()
    except Exception as exc:
        logger.exception(exc)


class VideoVisionTrack(aiortc.MediaStreamTrack):
    """A video stream track which performs computer vision work on video
    and returns video with debug info.
    """

    kind = "video"

    def __init__(
        self,
        input_track: aiortc.MediaStreamTrack,
        vision_analyzer: vision.VisionAnalyzer,
        use_debug_video: bool = False,
    ):
        """A video stream track which performs computer vision work on video
        and returns video with debug info.

        Args:
            input_track (aiortc.MediaStreamTrack): Receiving video track
            vision_analyzer (VisionAnalyzer): Computer vision analyzer
            use_debug_video (bool, optional): Return debug video stream.
            Defaults to False.
        """

        super().__init__()

        self.input_track = input_track
        self.use_debug_video = use_debug_video
        self.vision_analyzer = vision_analyzer

    @classmethod
    def from_video_track(
        cls,
        track: aiortc.MediaStreamTrack,
        vision_analyzer: vision.VisionAnalyzer,
        use_debug_video: bool,
    ) -> VideoVisionTrack:
        """Create VideoAnalyzeTrack from a video track and start
        corresponding vision worker process.

        Args:
            input_track (aiortc.MediaStreamTrack): Receiving video track
            vision_analyzer (VisionAnalyzer): Computer vision analyzer
            use_debug_video (bool, optional): Return debug video stream.
            Defaults to False.

        Returns:
            VideoAnalyzeTrack
        """

        track = media_relay.subscribe(track)
        return cls(track, vision_analyzer, use_debug_video)

    async def recv(self):
        video_frame: av_vframe.VideoFrame = await self.input_track.recv()  # type: ignore
        if video_frame.pts is None:
            raise AppException("Video frame has no frame count.")

        event = asyncio.get_event_loop()

        debug_video_ndarray = await event.run_in_executor(
            None,
            self.vision_analyzer.analyze_video_frame,
            video_frame.to_ndarray(format="bgr24"),
            self.use_debug_video,
        )

        if debug_video_ndarray is not None:
            return_video_frame = av_vframe.VideoFrame.from_ndarray(
                debug_video_ndarray, format="bgr24"
            )
        else:
            return_video_frame = video_frame

        return_video_frame.pts = video_frame.pts
        return_video_frame.time_base = video_frame.time_base
        return return_video_frame
