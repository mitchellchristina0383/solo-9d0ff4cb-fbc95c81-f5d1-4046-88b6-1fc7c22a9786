"""Real-socket test wiring.

ASGITransport cannot model client disconnect / read timeout semantics for
long-lived streaming generators (the stream context cannot even be entered
while the app generator is parked), so the suite runs the actual uvicorn
servers on localhost ports — exactly what Docker Compose runs, just without
containers. Three servers are started once per test session and torn down at
the end.
"""
from __future__ import annotations

import socket
import time

import httpx
import pytest
import uvicorn

from gateway.app import GatewayConfig, create_app as create_gateway
from node.app import NodeConfig, create_app as create_node


READ_TIMEOUT = 0.4
EMIT_INTERVAL = 0.02


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerHarness:
    def __init__(self, app, name: str):
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = __import__("threading").Thread(
            target=self.server.run, name=f"uvicorn-{name}", daemon=True
        )

    def start(self):
        self.thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("test server failed to start")

    def stop(self):
        self.server.should_exit = True


class Harness:
    def __init__(self):
        self.node_a_h = ServerHarness(
            create_node(NodeConfig("node-a", emit_interval=EMIT_INTERVAL)), "node-a"
        )
        self.node_b_h = ServerHarness(
            create_node(NodeConfig("node-b", emit_interval=EMIT_INTERVAL)), "node-b"
        )
        self.node_a_url = self.node_a_h.base_url
        self.node_b_url = self.node_b_h.base_url
        self.gateway_h = ServerHarness(
            create_gateway(GatewayConfig(
                node_a_url=self.node_a_url,
                node_b_url=self.node_b_url,
                read_timeout=READ_TIMEOUT,
            )),
            "gateway",
        )

    def start(self):
        for h in (self.node_a_h, self.node_b_h, self.gateway_h):
            h.start()

    def stop(self):
        for h in (self.gateway_h, self.node_a_h, self.node_b_h):
            h.stop()

    def client(self, base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=base_url, timeout=10.0)


@pytest.fixture(scope="session")
def harness():
    h = Harness()
    h.start()
    yield h
    h.stop()


@pytest.fixture()
def registry(harness):
    return harness


@pytest.fixture()
def gateway_client(harness):
    return httpx.AsyncClient(base_url=harness.gateway_h.base_url, timeout=15.0)
