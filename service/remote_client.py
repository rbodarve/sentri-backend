"""Simulated third-party device: query the SSE API through its PUBLIC ngrok URL.

Stands in for a remote client (phone, other backend) so the full round trip
ngrok edge -> FastAPI -> agentic RAG -> FastAPI -> ngrok edge -> client can be
exercised from this machine. Requests go to the public https URL, not localhost,
so they traverse ngrok exactly as a real device's would.

For each query it prints every SSE event as it arrives (with elapsed time), then
checks the stream is what a client can rely on: `meta` first, no `error`, all
terminal events present with `done` last, and the replayed tokens equal
`done.response_text` (a withheld answer streams no tokens and carries a `notice`).
Exits 1 if any check fails.

Stdlib only (no deps a real client wouldn't have). Usage:
    python service/remote_client.py                       # URL auto-read from ngrok (:4040)
    python service/remote_client.py --url https://x.ngrok-free.app -q "question" -q "another"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

# One question per agent route (answer + withhold), taken from eval/eval_agentic.json.
DEFAULT_QUERIES = [
    "Which contractor was awarded contract 24AJ0052?",                    # simple
    "Who is the District Engineer for contracts 24BJ0005 and 24CC0265?",  # fanout
    "List all the projects in the database",                              # enumerate
    "Who is the District Engineer for contract 24ZZ9999?",                # withhold
]
TERMINAL = ["contract_ids", "sources", "graph", "parts", "done"]


def ngrok_public_url() -> str:
    """The tunnel's public https URL, read from ngrok's local inspection API."""
    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=5) as r:
        tunnels = json.load(r)["tunnels"]
    return next(t["public_url"] for t in tunnels if t["public_url"].startswith("https://"))


def stream_events(url: str, query: str, session_id: str):
    """POST /query and yield (event, data) per SSE frame as it arrives."""
    body = json.dumps({"query": query, "session_id": session_id}).encode()
    req = urllib.request.Request(f"{url}/query", data=body, method="POST",
                                 headers={"content-type": "application/json",
                                          "accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=600) as r:
        event, data = None, []
        for raw in r:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif not line and event:
                yield event, json.loads("\n".join(data))
                event, data = None, []


def check(events: list[tuple[str, object]]) -> list[str]:
    """What a client relies on; returns the violations (empty = OK)."""
    names = [e for e, _ in events]
    problems = []
    if not names or names[0] != "meta":
        problems.append("first event is not `meta`")
    if "error" in names:
        problems.append(f"`error` event: {dict(events)['error']}")
    missing = [t for t in TERMINAL if t not in names]
    if missing:
        problems.append(f"missing terminal event(s): {missing}")
    if names and names[-1] != "done":
        problems.append(f"last event is `{names[-1]}`, not `done`")
    if "done" in names:
        done = dict(events)["done"]
        tokens = "".join(d["text"] for e, d in events if e == "token")
        if tokens != done["response_text"]:
            problems.append("replayed tokens != done.response_text")
        if done["uncertain"] and not done["notice"]:
            problems.append("withheld answer without a `notice`")
    return problems


def run(url: str, query: str, session_id: str) -> bool:
    print(f"\n>>> {query}")
    events, t0 = [], time.monotonic()
    for event, data in stream_events(url, query, session_id):
        events.append((event, data))
        if event != "token":  # tokens are shown once, joined, below
            print(f"  [{time.monotonic() - t0:6.1f}s] {event}: "
                  f"{json.dumps(data, ensure_ascii=False)[:300]}")
    done = dict(events).get("done", {})
    print(f"  answer: {done.get('response_text') or '(withheld) ' + str(done.get('notice'))}")
    problems = check(events)
    print("  CHECK: " + ("OK" if not problems else "FAIL -- " + "; ".join(problems)))
    return not problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", help="public base URL (default: read from ngrok on :4040)")
    ap.add_argument("-q", "--query", action="append", help="question to ask (repeatable)")
    ap.add_argument("--session", default="remote-sim", help="session_id sent with each query")
    args = ap.parse_args()

    url = (args.url or ngrok_public_url()).rstrip("/")
    with urllib.request.urlopen(f"{url}/health", timeout=30) as r:
        print(f"{url}/health -> {r.read().decode()}")
    results = [run(url, q, args.session) for q in (args.query or DEFAULT_QUERIES)]
    print(f"\n{sum(results)}/{len(results)} streams OK via {url}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
