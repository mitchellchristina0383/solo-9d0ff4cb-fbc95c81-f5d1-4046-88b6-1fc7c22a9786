"""End-to-end behavior tests for the streaming failover gateway.

Run from the repository root:  python -m pytest -v

Tests are written as async coroutines and driven with asyncio.run (no extra
pytest plugin required).
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from common.deterministic import expected_token_ids, token_text


PROMPT = "请写一段关于秋天的短文"
MAX_TOKENS = 12
STALL_BEFORE = 4


async def collect_sse(client: httpx.AsyncClient, scenario: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async with client.stream(
        "POST", "/api/chat",
        json={"prompt": PROMPT, "max_tokens": MAX_TOKENS, "scenario": scenario},
    ) as resp:
        assert resp.status_code == 200
        event = None
        data_lines: list[str] = []
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if event is not None or data_lines:
                    events.append((event or "message", json.loads("\n".join(data_lines))))
                event, data_lines = None, []
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if event is not None or data_lines:
            events.append((event or "message", json.loads("\n".join(data_lines))))
    return events


def tokens_of(events):
    return [(e["index"], e["id"], e["text"], e["node"]) for name, e in events if name == "token"]


def names_of(events):
    return [name for name, _ in events]


# --- async scenarios ---------------------------------------------------------

async def _healthy(gateway_client):
    events = await collect_sse(gateway_client, "healthy")
    tokens = tokens_of(events)
    assert len(tokens) == MAX_TOKENS
    assert [t[0] for t in tokens] == list(range(MAX_TOKENS))           # strict order
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, MAX_TOKENS)
    assert [t[3] for t in tokens] == ["node-a"] * MAX_TOKENS          # only primary
    assert names_of(events)[-1] == "done"


async def _failover_success(gateway_client):
    healthy = tokens_of(await collect_sse(gateway_client, "healthy"))
    events = await collect_sse(gateway_client, "failover_success")
    names = names_of(events)

    assert "fallback" in names
    fallback = next(e for n, e in events if n == "fallback")
    assert fallback["reason"] == "read_timeout"
    assert fallback["accepted_count"] == STALL_BEFORE

    tokens = tokens_of(events)
    assert len(tokens) == MAX_TOKENS
    assert [t[0] for t in tokens] == list(range(MAX_TOKENS))          # no gaps/dup
    # Append-only: first STALL_BEFORE tokens from primary, rest from backup.
    assert [t[3] for t in tokens] == ["node-a"] * STALL_BEFORE + ["node-b"] * (MAX_TOKENS - STALL_BEFORE)
    # The decisive property: continuation reproduces the healthy stream.
    assert [(i, tid, text) for i, tid, text, _ in tokens] == \
           [(i, tid, text) for i, tid, text, _ in healthy]
    done = next(e for n, e in events if n == "done")
    assert done["identical_to_healthy"] is True
    assert done["nodes"] == {"node-a": STALL_BEFORE, "node-b": MAX_TOKENS - STALL_BEFORE}


async def _shared_budget(gateway_client):
    events = await collect_sse(gateway_client, "failover_success")
    done = next(e for n, e in events if n == "done")
    assert done["token_count"] == MAX_TOKENS
    assert len(tokens_of(events)) == MAX_TOKENS


async def _backup_failure(gateway_client):
    events = await collect_sse(gateway_client, "failover_backup_error")
    tokens = tokens_of(events)
    # Primary produced 4, backup fails at absolute index 7 -> 3 more shown.
    assert len(tokens) == 7
    assert [t[3] for t in tokens] == ["node-a"] * 4 + ["node-b"] * 3
    incomplete = [e for n, e in events if n == "incomplete"]
    assert len(incomplete) == 1
    assert incomplete[0]["reason"] == "backup_failed"
    # Displayed text is the untouched deterministic prefix.
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, 7)
    assert incomplete[0]["prefix_text"] == " ".join(token_text(PROMPT, i) for i in range(7))
    assert "done" not in names_of(events)


async def _version_mismatch(gateway_client):
    events = await collect_sse(gateway_client, "version_mismatch")
    tokens = tokens_of(events)
    assert len(tokens) == STALL_BEFORE
    assert all(t[3] == "node-a" for t in tokens)
    incomplete = next(e for n, e in events if n == "incomplete")
    assert incomplete["reason"] == "version_mismatch"
    assert incomplete["backup_identity"]["tokenizer_version"] == "tok-v2"
    assert incomplete["primary_identity"]["tokenizer_version"] == "tok-v1"
    # Prefix preserved verbatim.
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, STALL_BEFORE)
    assert "done" not in names_of(events)


async def _late_packets_ignored(registry, gateway_client):
    events = await collect_sse(gateway_client, "failover_success")
    tokens = tokens_of(events)

    # The stalled node's detached late-generation task recorded late packets.
    client = registry.client(registry.node_a_url)
    async with client:
        late = (await client.get("/debug/late")).json()
    late_entries = [v for values in late.values() for v in values]
    assert late_entries, "primary should have emitted late packets after timeout"
    late_indices = {p["index"] for p in late_entries}
    # Every late index was accepted exactly once (from the backup), never via
    # the late primary delivery.
    accepted = {t[0]: t[3] for t in tokens}
    for idx in late_indices:
        assert accepted[idx] == "node-b"
    # No duplicate indices in the user-visible stream.
    assert len({t[0] for t in tokens}) == len(tokens)


async def _backup_rejects_wrong_prefix(registry):
    client = registry.client(registry.node_b_url)
    async with client:
        r = await client.post("/generate", json={
            "request_id": "bad", "prompt": PROMPT, "start": 2,
            "max_tokens": 3, "accepted_ids": [1, 1],
        })
        assert r.status_code == 409


async def _determinism(registry):
    async def fetch(url):
        client = registry.client(url)
        async with client:
            r = await client.post("/generate", json={
                "request_id": "x", "prompt": PROMPT, "start": 0, "max_tokens": 5,
            })
            ids = []
            async for line in r.aiter_lines():
                if line.startswith("data:") and '"id"' in line:
                    ids.append(json.loads(line[5:].lstrip())["id"])
            return ids
    a = await fetch(registry.node_a_url)
    b1 = await fetch(registry.node_b_url)
    b2 = await fetch(registry.node_b_url)
    assert a == b1 == b2 == expected_token_ids(PROMPT, 0, 5)


# --- sync wrappers -----------------------------------------------------------

def test_healthy_baseline_matches_deterministic_model(gateway_client):
    asyncio.run(_healthy(gateway_client))


def test_failover_success_is_identical_to_healthy(gateway_client):
    asyncio.run(_failover_success(gateway_client))


def test_shared_token_budget_never_exceeded(gateway_client):
    asyncio.run(_shared_budget(gateway_client))


def test_backup_failure_keeps_prefix_and_marks_incomplete(gateway_client):
    asyncio.run(_backup_failure(gateway_client))


def test_version_mismatch_refuses_continuation(gateway_client):
    asyncio.run(_version_mismatch(gateway_client))


def test_late_packets_from_primary_are_ignored(registry, gateway_client):
    asyncio.run(_late_packets_ignored(registry, gateway_client))


def test_backup_rejects_wrong_prefix(registry):
    asyncio.run(_backup_rejects_wrong_prefix(registry))


def test_determinism_across_nodes_and_runs(registry):
    asyncio.run(_determinism(registry))
