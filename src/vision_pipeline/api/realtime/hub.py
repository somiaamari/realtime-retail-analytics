"""Bounded per-client WebSocket fan-out hub."""

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class HubClient:
    """One stream WebSocket subscriber."""

    stream_id: str
    queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=32)
    )


class WebSocketHub:
    """Manage client subscriptions and drop stale queued messages safely."""

    def __init__(self) -> None:
        self._clients: dict[str, set[HubClient]] = {}

    def subscribe(self, stream_id: str, max_clients: int) -> HubClient:
        """Create a client subscription or reject it when at capacity."""
        clients = self._clients.setdefault(stream_id, set())
        if sum(len(group) for group in self._clients.values()) >= max_clients:
            raise RuntimeError("Maximum WebSocket client count reached")
        client = HubClient(stream_id)
        clients.add(client)
        return client

    def unsubscribe(self, client: HubClient) -> None:
        """Remove a client and its stream entry if empty."""
        clients = self._clients.get(client.stream_id)
        if clients is not None:
            clients.discard(client)
            if not clients:
                self._clients.pop(client.stream_id, None)

    def publish(self, stream_id: str, message: dict[str, Any]) -> None:
        """Fan out one message without waiting on any client."""
        for client in tuple(self._clients.get(stream_id, ())):
            if client.queue.full():
                try:
                    client.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                client.queue.put_nowait(message)
            except asyncio.QueueFull:
                continue

    @property
    def client_count(self) -> int:
        """Return the number of active WebSocket clients."""
        return sum(len(clients) for clients in self._clients.values())
