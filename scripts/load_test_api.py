"""Exercise a demo API with REST polling, WebSockets, and MJPEG clients."""

import argparse
import asyncio
import time

import httpx
import websockets


async def run_load(base_url: str, duration: float) -> None:
    """Run concurrent clients and print successful operation totals."""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    counts = {"websocket_messages": 0, "mjpeg_chunks": 0, "rest_requests": 0}
    stop_at = time.monotonic() + duration

    async def websocket_client() -> None:
        async with websockets.connect(f"{ws_url}/ws/streams/demo") as socket:
            while time.monotonic() < stop_at:
                await asyncio.wait_for(socket.recv(), timeout=5)
                counts["websocket_messages"] += 1

    async def mjpeg_client() -> None:
        async with httpx.AsyncClient(timeout=duration + 10) as client:
            async with client.stream(
                "GET", f"{base_url}/api/v1/streams/demo/video"
            ) as response:
                async for chunk in response.aiter_bytes():
                    if time.monotonic() >= stop_at:
                        return
                    if chunk:
                        counts["mjpeg_chunks"] += 1

    async def rest_poll() -> None:
        async with httpx.AsyncClient() as client:
            while time.monotonic() < stop_at:
                response = await client.get(
                    f"{base_url}/api/v1/streams/demo/analytics/summary"
                )
                response.raise_for_status()
                counts["rest_requests"] += 1
                await asyncio.sleep(0.25)

    tasks = [
        *(asyncio.create_task(websocket_client()) for _ in range(50)),
        *(asyncio.create_task(mjpeg_client()) for _ in range(5)),
        asyncio.create_task(rest_poll()),
    ]
    await asyncio.sleep(duration)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print({"duration_seconds": duration, **counts})


def main() -> None:
    """Parse load test options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()
    asyncio.run(run_load(args.url.rstrip("/"), args.duration))


if __name__ == "__main__":
    main()
