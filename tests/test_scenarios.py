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
    out = []
    for name, e in events:
        if name == "token":
            out.append((e["index"], e["id"], e["text"], e["node"]))
        elif name == "token_batch":
            for t in e["tokens"]:
                out.append((t["index"], t["id"], t["text"], t["node"]))
    return out


def token_batches_of(events):
    return [list(e["tokens"]) for n, e in events if n == "token_batch"]


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


# --- token batching ----------------------------------------------------------

async def _batcher_unit():
    from gateway.app import TokenBatcher

    b = TokenBatcher(size=3, max_wait=0.08)

    def tok(index, node="node-a"):
        return {"index": index, "id": index * 10, "text": f"t{index}", "node": node}

    # The first two tokens stay buffered (size not reached yet).
    await b.add(tok(0))
    await b.add(tok(1))
    assert b.pending()
    # Third token fills the batch: the orchestrator then flushes immediately.
    await b.add(tok(2))
    assert b.is_full()
    full = await b.flush()
    assert [t["index"] for t in full] == [0, 1, 2]
    assert not b.pending()
    await b.close()


async def _batcher_max_wait():
    from gateway.app import TokenBatcher

    b = TokenBatcher(size=3, max_wait=0.05)

    def tok(index):
        return {"index": index, "id": index, "text": f"t{index}", "node": "node-a"}

    await b.add(tok(0))
    # No further token arrives (upstream silence): the deadline still fires.
    deadline = b.timer()
    armed_epoch = b.epoch
    done, _ = await asyncio.wait({deadline}, timeout=0.5)
    assert deadline in done
    assert b.epoch == armed_epoch  # no spurious re-arm while waiting
    batch = await b.flush()
    assert [t["index"] for t in batch] == [0]
    assert not b.pending()
    await b.close()


async def _batcher_explicit_flush():
    from gateway.app import TokenBatcher

    b = TokenBatcher(size=3, max_wait=10)

    def tok(index):
        return {"index": index, "id": index, "text": f"t{index}", "node": "node-a"}

    await b.add(tok(0))
    await b.add(tok(1))
    # End of reply / node switch: the pending partial batch must be sent now.
    batch = await b.flush()
    assert [t["index"] for t in batch] == [0, 1]
    # Flushing with an empty buffer is a no-op (never emits an empty event).
    assert await b.flush() is None
    await b.close()


async def _batcher_node_switch_splits_batch():
    from gateway.app import TokenBatcher

    b = TokenBatcher(size=3, max_wait=10)

    def tok(index, node):
        return {"index": index, "id": index, "text": f"t{index}", "node": node}

    await b.add(tok(0, "node-a"))
    await b.add(tok(1, "node-a"))
    # A token from another node flushes the old batch before joining a new one.
    forced = await b.add(tok(2, "node-b"))
    assert [(t["index"], t["node"]) for t in forced] == [(0, "node-a"), (1, "node-a")]
    # The new node's token is the only thing left buffered.
    second = await b.flush()
    assert [(t["index"], t["node"]) for t in second] == [(2, "node-b")]
    assert await b.flush() is None
    await b.close()


async def _healthy_batching(gateway_client):
    events = await collect_sse(gateway_client, "healthy")
    names = names_of(events)
    assert "token" not in names, "gateway must send token_batch, not bare token events"
    assert names[0] == "meta"
    assert names[-1] == "done"
    batches = token_batches_of(events)
    # Every batch is ordered, has no gaps/duplicates and never exceeds size 3.
    flat_indices = [t["index"] for batch in batches for t in batch]
    assert flat_indices == list(range(MAX_TOKENS))
    assert all(len(batch) <= 3 for batch in batches)
    # 12 tokens with size 3: four full batches.
    assert [len(batch) for batch in batches] == [3, 3, 3, 3]
    for batch in batches:
        assert all(t["node"] == "node-a" for t in batch)
        assert all("id" in t and "text" in t for t in batch)


async def _failover_batching(gateway_client):
    events = await collect_sse(gateway_client, "failover_success")
    names = names_of(events)
    fallback_pos = names.index("fallback")

    primary_batches = [e for n, e in events[:fallback_pos] if n == "token_batch"]
    # The 4 primary tokens were all delivered *before* the switch (the last
    # partial batch is flushed by the max-wait deadline or the switch itself).
    primary_tokens = [t for batch in primary_batches for t in batch["tokens"]]
    assert len(primary_tokens) == STALL_BEFORE
    assert all(t["node"] == "node-a" for t in primary_tokens)

    backup_batches = [e for n, e in events[fallback_pos + 1:] if n == "token_batch"]
    backup_tokens = [t for batch in backup_batches for t in batch["tokens"]]
    assert len(backup_tokens) == MAX_TOKENS - STALL_BEFORE
    assert all(t["node"] == "node-b" for t in backup_tokens)

    # Every batch belongs to exactly one node and keeps the global order.
    for event in (primary_batches + backup_batches):
        nodes = {t["node"] for t in event["tokens"]}
        assert len(nodes) == 1
        assert len(event["tokens"]) <= 3
    assert tokens_of(events)[-1][0] == MAX_TOKENS - 1
    assert names[-1] == "done"


async def _backup_failure_batching(gateway_client):
    events = await collect_sse(gateway_client, "failover_backup_error")
    batches = token_batches_of(events)
    assert all(len(batch) <= 3 for batch in batches)
    # All 7 accepted tokens still reach the page despite the failure.
    assert [t["index"] for batch in batches for t in batch] == list(range(7))
    # The incomplete event comes after the final flushed batch.
    names = names_of(events)
    assert names[-1] == "incomplete"
    incomplete = next(e for n, e in events if n == "incomplete")
    assert incomplete["accepted_count"] == 7


def test_token_batcher_size_flush():
    asyncio.run(_batcher_unit())


def test_token_batcher_max_wait_deadline_flushes_on_silence():
    asyncio.run(_batcher_max_wait())


def test_token_batcher_explicit_flush_before_switch_or_end():
    asyncio.run(_batcher_explicit_flush())


def test_token_batcher_never_mixes_nodes_in_one_batch():
    asyncio.run(_batcher_node_switch_splits_batch())


def test_healthy_stream_is_sent_as_ordered_token_batches(gateway_client):
    asyncio.run(_healthy_batching(gateway_client))


def test_failover_flushes_primary_batch_before_switch(gateway_client):
    asyncio.run(_failover_batching(gateway_client))


def test_backup_failure_still_delivers_every_accepted_batch(gateway_client):
    asyncio.run(_backup_failure_batching(gateway_client))
