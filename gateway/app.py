"""Streaming failover gateway.

Responsibilities:

* Serve a browser page and an SSE ``/api/chat`` stream.
* Forward generation to a primary mock node with a *per-read* timeout.
* On mid-stream timeout (or a mid-stream node error), continue on the backup
  node using the original prompt plus the tokens already accepted by the
  gateway (the user has already seen them).
* Append-only state machine: each absolute token index is accepted at most
  once; any late packet from the abandoned primary is ignored by index.
* One shared token budget for the whole reply: the backup only gets
  ``max_tokens - accepted_count``.
* Refuse failover (keep displayed text, mark incomplete) when the two nodes
  report different model/tokenizer identities, or when the backup fails.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common.deterministic import expected_token_ids, token_text

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


# ---------------------------------------------------------------------------
# Scenario catalog: repeatable fault injection, applied by the gateway only.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeFault:
    """Fault instructions sent to one mock node for one request."""
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    info_headers: dict[str, str] = field(default_factory=dict)


SCENARIOS: dict[str, dict[str, NodeFault]] = {
    # No fault: the primary streams the whole reply.
    "healthy": {"primary": NodeFault(), "backup": NodeFault()},
    # Primary stalls mid-stream -> read timeout -> backup continues and the
    # final reply is identical to the healthy run.
    "failover_success": {
        "primary": NodeFault(
            headers={"X-Mock-Stall-Seconds": "30"},
            body={"stall_before": 4},
        ),
        "backup": NodeFault(),
    },
    # Primary times out, backup itself fails mid-continuation -> keep the
    # displayed prefix and mark the reply incomplete.
    "failover_backup_error": {
        "primary": NodeFault(
            headers={"X-Mock-Stall-Seconds": "30"},
            body={"stall_before": 4},
        ),
        "backup": NodeFault(body={"fail_at": 7}),
    },
    # Primary times out, backup reports a different model/tokenizer version
    # -> no continuation, keep prefix, mark incomplete.
    "version_mismatch": {
        "primary": NodeFault(
            headers={"X-Mock-Stall-Seconds": "30"},
            body={"stall_before": 4},
        ),
        "backup": NodeFault(
            headers={"X-Mock-Model-Override": "mock-llm-7b-hotfix",
                     "X-Mock-Tokenizer-Override": "tok-v2"},
            info_headers={"X-Mock-Model-Override": "mock-llm-7b-hotfix",
                          "X-Mock-Tokenizer-Override": "tok-v2"},
        ),
    },
}


@dataclass(frozen=True)
class GatewayConfig:
    node_a_url: str = os.environ.get("NODE_A_URL", "http://node-a:8000")
    node_b_url: str = os.environ.get("NODE_B_URL", "http://node-b:8000")
    primary_name: str = "node-a"
    backup_name: str = "node-b"
    # Maximum silence (no SSE event) before the primary is declared timed out.
    read_timeout: float = float(os.environ.get("GATEWAY_READ_TIMEOUT", "1.0"))
    web_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_tokens: int = Field(ge=1, le=128)
    scenario: str = "healthy"


def sse(event: str, data: dict) -> bytes:
    lines = "".join(f"data: {line}\n" for line in json.dumps(data, ensure_ascii=False).split("\n"))
    return f"event: {event}\n{lines}\n".encode()


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Parse an httpx SSE stream into (event, data-string) pairs."""
    event: str | None = None
    data_lines: list[str] = []
    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if line == "":
            if event is not None or data_lines:
                yield event or "message", "\n".join(data_lines)
            event, data_lines = None, []
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    if event is not None or data_lines:
        yield event or "message", "\n".join(data_lines)


ClientFactory = Callable[[str], Awaitable[httpx.AsyncClient]]


def default_client_factory(base_url: str) -> httpx.AsyncClient:
    # Generous client timeouts; the gateway enforces its own read timeout so
    # the behavior is identical with in-process test transports.
    return httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(30.0, read=30.0))


def create_app(
    config: GatewayConfig | None = None,
    client_factory: Callable[[str], httpx.AsyncClient] | None = None,
) -> FastAPI:
    cfg = config or GatewayConfig()
    make_client = client_factory or default_client_factory
    app = FastAPI(title="streaming-failover-gateway")

    @app.get("/api/scenarios")
    async def scenarios() -> dict:
        return {"scenarios": [
            {"id": sid, "primary": dict(headers=faults["primary"].headers, body=faults["primary"].body),
             "backup": dict(headers=faults["backup"].headers, body=faults["backup"].body)}
            for sid, faults in SCENARIOS.items()
        ]}

    @app.get("/api/expected")
    async def expected(prompt: str, max_tokens: int) -> dict:
        """Ground-truth token stream of the deterministic mock model."""
        ids = expected_token_ids(prompt, 0, max_tokens)
        return {"ids": ids, "texts": [token_text(prompt, i) for i in range(max_tokens)]}

    @app.post("/api/chat")
    async def chat(body: ChatRequest) -> StreamingResponse:
        if body.scenario not in SCENARIOS:
            return StreamingResponse(
                iter([sse("incomplete", {"reason": "unknown_scenario", "scenario": body.scenario})]),
                media_type="text/event-stream",
            )
        faults = SCENARIOS[body.scenario]
        request_id = uuid.uuid4().hex
        return StreamingResponse(
            _orchestrate(body, request_id, faults, cfg, make_client),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.mount("/", StaticFiles(directory=cfg.web_dir, html=True), name="web")
    return app


# ---------------------------------------------------------------------------
# Orchestration state machine
# ---------------------------------------------------------------------------

async def _orchestrate(
    body: ChatRequest,
    request_id: str,
    faults: dict[str, NodeFault],
    cfg: GatewayConfig,
    make_client: Callable[[str], httpx.AsyncClient],
) -> AsyncIterator[bytes]:
    prompt, max_tokens = body.prompt, body.max_tokens
    accepted_ids: list[int] = []
    accepted_texts: list[str] = []
    provenance: list[str] = []
    primary_identity: tuple[str, str] | None = None
    next_index = 0

    yield sse("meta", {
        "request_id": request_id,
        "scenario": body.scenario,
        "max_tokens": max_tokens,
        "primary": cfg.primary_name,
        "backup": cfg.backup_name,
    })

    def accept_token(obj: dict, node: str) -> bool:
        """Append-only acceptance: one token per absolute index, in order."""
        nonlocal next_index
        index = int(obj["index"])
        if index != next_index:
            logger.warning(
                "ignoring out-of-order/late token index=%s (next expected=%s) from %s",
                index, next_index, node,
            )
            return False
        accepted_ids.append(int(obj["id"]))
        accepted_texts.append(str(obj["text"]))
        provenance.append(node)
        next_index += 1
        return True

    # ---- primary node ------------------------------------------------------
    primary_fault = faults["primary"]
    primary_payload = {
        "request_id": request_id,
        "prompt": prompt,
        "start": 0,
        "max_tokens": max_tokens,
        "accepted_ids": [],
        "prefix_text": "",
        **primary_fault.body,
    }

    timed_out = False
    primary_error: str | None = None
    try:
        async with make_client(cfg.node_a_url) as client:
            async with client.stream("POST", "/generate", json=primary_payload,
                                     headers=primary_fault.headers) as resp:
                if resp.status_code != 200:
                    detail = await resp.aread()
                    primary_error = f"http_{resp.status_code}:{detail[:200]!r}"
                else:
                    event_iter = iter_sse_events(resp)
                    while True:
                        try:
                            event, data = await asyncio.wait_for(
                                event_iter.__anext__(), timeout=cfg.read_timeout
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            timed_out = True
                            logger.warning(
                                "primary %s read timeout after %d accepted token(s) for %s",
                                cfg.primary_name, next_index, request_id,
                            )
                            break
                        if event == "meta":
                            info = json.loads(data)
                            primary_identity = (info["model_id"], info["tokenizer_version"])
                        elif event == "token":
                            obj = json.loads(data)
                            if accept_token(obj, cfg.primary_name):
                                yield sse("token", {**obj, "node": cfg.primary_name})
                                if next_index >= max_tokens:
                                    break
                        elif event == "error":
                            primary_error = data
                            logger.warning("primary %s stream error: %s", cfg.primary_name, data)
                            break
                        elif event == "done":
                            break
    except httpx.HTTPError as exc:
        primary_error = f"{type(exc).__name__}: {exc}"

    # Healthy completion on the primary: nothing else to do.
    if not timed_out and primary_error is None and next_index >= max_tokens:
        yield sse("done", {
            "reason": "max_tokens", "token_count": next_index,
            "nodes": _node_shares(provenance),
        })
        return

    # Primary finished without error but short of the budget: treat as failure.
    if not timed_out and primary_error is None:
        primary_error = "primary_ended_early"

    if next_index >= max_tokens:
        yield sse("done", {
            "reason": "max_tokens", "token_count": next_index,
            "nodes": _node_shares(provenance),
        })
        return

    yield sse("fallback", {
        "reason": "read_timeout" if timed_out else "primary_error",
        "from_node": cfg.primary_name,
        "to_node": cfg.backup_name,
        "accepted_count": next_index,
        "detail": primary_error,
    })

    # ---- backup node: identity check --------------------------------------
    backup_fault = faults["backup"]
    async with make_client(cfg.node_b_url) as client:
        try:
            info_resp = await client.get("/info", headers=backup_fault.info_headers)
            info_resp.raise_for_status()
            backup_info = info_resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            yield sse("incomplete", {
                "reason": "backup_unreachable",
                "detail": str(exc),
                "accepted_count": next_index,
                "prefix_text": " ".join(accepted_texts),
            })
            return
        backup_identity = (backup_info["model_id"], backup_info["tokenizer_version"])

        if primary_identity is not None and backup_identity != primary_identity:
            logger.warning(
                "refusing failover for %s: identity mismatch primary=%s backup=%s",
                request_id, primary_identity, backup_identity,
            )
            yield sse("incomplete", {
                "reason": "version_mismatch",
                "primary_identity": {"model_id": primary_identity[0],
                                     "tokenizer_version": primary_identity[1]},
                "backup_identity": {"model_id": backup_identity[0],
                                    "tokenizer_version": backup_identity[1]},
                "accepted_count": next_index,
                "prefix_text": " ".join(accepted_texts),
            })
            return

        # ---- backup continuation with the shared token budget -------------
        remaining = max_tokens - next_index
        backup_payload = {
            "request_id": request_id,
            "prompt": prompt,
            "start": next_index,
            "max_tokens": remaining,
            "accepted_ids": accepted_ids,
            "prefix_text": " ".join(accepted_texts),
            **backup_fault.body,
        }
        backup_failed: str | None = None
        try:
            async with client.stream("POST", "/generate", json=backup_payload,
                                     headers=backup_fault.headers) as resp:
                if resp.status_code != 200:
                    detail = await resp.aread()
                    backup_failed = f"http_{resp.status_code}:{detail[:200]!r}"
                else:
                    async for event, data in iter_sse_events(resp):
                        if event == "token":
                            obj = json.loads(data)
                            if accept_token(obj, cfg.backup_name):
                                yield sse("token", {**obj, "node": cfg.backup_name})
                                if next_index >= max_tokens:
                                    break
                        elif event == "error":
                            backup_failed = data
                            break
                        elif event == "done":
                            break
        except httpx.HTTPError as exc:
            backup_failed = f"{type(exc).__name__}: {exc}"

    if backup_failed is not None or next_index < max_tokens:
        logger.warning("backup continuation failed for %s: %s", request_id, backup_failed)
        yield sse("incomplete", {
            "reason": "backup_failed",
            "detail": backup_failed,
            "accepted_count": next_index,
            "prefix_text": " ".join(accepted_texts),
        })
        return

    yield sse("done", {
        "reason": "max_tokens",
        "token_count": next_index,
        "nodes": _node_shares(provenance),
        "identical_to_healthy": True,
    })


def _node_shares(provenance: list[str]) -> dict[str, int]:
    shares: dict[str, int] = {}
    for node in provenance:
        shares[node] = shares.get(node, 0) + 1
    return shares


app = create_app()
