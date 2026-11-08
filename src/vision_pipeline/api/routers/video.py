"""MJPEG and single-frame video endpoints."""

import threading
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

router = APIRouter(prefix="/streams/{stream_id}", tags=["video"])


@router.get("/frame.jpg")
def frame_jpg(stream_id: str, request: Request) -> Response:
    """Return the most recently annotated JPEG frame."""
    pipeline = request.app.state.stream_manager.get_pipeline(stream_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Stream is not running.")
    frame = pipeline.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame is available yet.")
    return Response(frame.jpeg, media_type="image/jpeg")


@router.get("/video")
def video_mjpeg(stream_id: str, request: Request) -> StreamingResponse:
    """Stream annotated JPEG frames using multipart/x-mixed-replace."""
    manager = request.app.state.stream_manager
    pipeline = manager.get_pipeline(stream_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Stream is not running.")
    guard: threading.Lock = request.app.state.mjpeg_guard
    with guard:
        if (
            request.app.state.mjpeg_clients
            >= request.app.state.settings.max_mjpeg_clients
        ):
            raise HTTPException(status_code=503, detail="MJPEG client limit reached.")
        request.app.state.mjpeg_clients += 1

    def generate() -> Iterator[bytes]:
        version = 0
        try:
            while True:
                frame = pipeline.wait_for_frame(version, timeout=1.0)
                if frame is None:
                    continue

                version = frame.version

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame.jpeg + b"\r\n"
                )
        finally:
            with guard:
                request.app.state.mjpeg_clients -= 1
    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
