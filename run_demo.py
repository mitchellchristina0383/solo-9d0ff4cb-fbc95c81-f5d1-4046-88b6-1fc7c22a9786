"""Real-HTTP trial run of every scenario (no Docker required).

Spins up the actual uvicorn servers for node-a, node-b and the gateway on
localhost ports, then drives the SSE API exactly like the browser does and
prints a full report: accepted tokens per node, failover events, the late
packets the abandoned primary emitted afterwards, and the equivalence check
against the healthy baseline.

Usage:  python run_demo.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time

import httpx
import uvicorn

from common.deterministic import expected_token_ids
from gateway.app import GatewayConfig, create_app as create_gateway
from node.app import NodeConfig, create_app as create_node

PROMPT = "请写一段关于秋天的短文"
MAX_TOKENS = 12
PORT_A, PORT_B, PORT_G = 8901, 8902, 8908
READ_TIMEOUT = 0.5
EMIT_INTERVAL = 0.05


def serve(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


async def consume(client: httpx.AsyncClient, scenario: str):
    events: list[tuple[str, dict]] = []
    async with client.stream(
        "POST", "http://127.0.0.1:8908/api/chat",
        json={"prompt": PROMPT, "max_tokens": MAX_TOKENS, "scenario": scenario},
    ) as resp:
        event = None
        data: list[str] = []
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if event is not None or data:
                    events.append((event or "message", json.loads("\n".join(data))))
                event, data = None, []
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
    return events


def report(scenario: str, events, healthy_ids):
    print(f"\n{'=' * 78}\n场景: {scenario}\n{'=' * 78}")
    tokens = [e for n, e in events if n == "token"]
    for name, e in events:
        if name == "meta":
            print(f"[meta] request={e['request_id'][:8]}… 上限={e['max_tokens']}")
        elif name == "fallback":
            print(f"[fallback] {e['reason']}：{e['from_node']} 已采纳 {e['accepted_count']} "
                  f"个词元后切换到 {e['to_node']}")
        elif name == "incomplete":
            print(f"[incomplete] reason={e['reason']} 保留 {e['accepted_count']} 个词元")
            if "primary_identity" in e:
                print(f"             primary={e['primary_identity']}")
                print(f"             backup ={e['backup_identity']}")
    by_node: dict[str, list[str]] = {}
    for t in tokens:
        by_node.setdefault(t["node"], []).append(t["text"])
    for node, texts in by_node.items():
        print(f"[tokens:{node}] ({len(texts)}) {' '.join(texts)}")
    ids = [t["id"] for t in tokens]
    final = "done" in {n for n, _ in events}
    if final:
        ok = ids == healthy_ids
        print(f"[核对] 完整回复词元数={len(ids)}/{MAX_TOKENS}，与 healthy 一致: {ok}")
        assert ok, "failover result diverged from healthy baseline"
    else:
        n = len(ids)
        ok = ids == healthy_ids[:n]
        print(f"[核对] 未完成，保留前缀 {n}/{MAX_TOKENS} 个词元，与 healthy 同位置前缀一致: {ok}")
        assert ok
    return tokens


async def main():
    node_a = create_node(NodeConfig("node-a", emit_interval=EMIT_INTERVAL))
    node_b = create_node(NodeConfig("node-b", emit_interval=EMIT_INTERVAL))
    gateway = create_gateway(GatewayConfig(
        node_a_url=f"http://127.0.0.1:{PORT_A}",
        node_b_url=f"http://127.0.0.1:{PORT_B}",
        read_timeout=READ_TIMEOUT,
    ))
    servers = [serve(node_a, PORT_A), serve(node_b, PORT_B), serve(gateway, PORT_G)]
    # Wait for readiness.
    async with httpx.AsyncClient() as client:
        for port in (PORT_A, PORT_B, PORT_G):
            for _ in range(50):
                try:
                    r = await client.get(f"http://127.0.0.1:{port}/info" if port != PORT_G
                                         else f"http://127.0.0.1:{port}/api/scenarios")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.1)

    healthy_ids = expected_token_ids(PROMPT, 0, MAX_TOKENS)
    print(f"确定性基准词元 id 序列: {healthy_ids}")
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            healthy = report("healthy", await consume(client, "healthy"), healthy_ids)
            report("failover_success", await consume(client, "failover_success"), healthy_ids)
            report("failover_backup_error", await consume(client, "failover_backup_error"), healthy_ids)
            report("version_mismatch", await consume(client, "version_mismatch"), healthy_ids)

            # Prove the abandoned primary really sent late packets afterwards.
            await asyncio.sleep(2.2)
            late = (await client.get(f"http://127.0.0.1:{PORT_A}/debug/late")).json()
            flat = [p for vals in late.values() for p in vals]
            print(f"\n[迟到回包] node-a 在被放弃后仍产生 {len(flat)} 个迟到词元 "
                  f"(indices={sorted({p['index'] for p in flat})})，网关均未采纳。")
            assert flat
    finally:
        for s in servers:
            s.should_exit = True
    print(f"\n全部场景试跑通过，耗时 {time.time() - started:.1f}s。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\n试跑失败: {exc}", file=sys.stderr)
        sys.exit(1)
