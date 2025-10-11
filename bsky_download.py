from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

BASE = "https://public.api.bsky.app/xrpc"


def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def resolve_profile(actor: str) -> Dict[str, Any]:
    return _get(f"{BASE}/app.bsky.actor.getProfile", {"actor": actor})


def get_author_feed(actor: str, cursor: Optional[str] = None, limit: int = 100, filter_: str = "posts_with_replies") -> Dict[str, Any]:
    params = {"actor": actor, "limit": min(100, max(1, int(limit))), "filter": filter_}
    if cursor:
        params["cursor"] = cursor
    return _get(f"{BASE}/app.bsky.feed.getAuthorFeed", params)


def get_post_thread(uri: str, depth: int = 100, parent_height: int = 100) -> Dict[str, Any]:
    return _get(
        f"{BASE}/app.bsky.feed.getPostThread",
        {"uri": uri, "depth": depth, "parentHeight": parent_height},
    )


def _collect_thread_text(node: Dict[str, Any], author_did: str, out: List[str]) -> None:
    if not isinstance(node, dict):
        return
    post = node.get("post")
    if isinstance(post, dict):
        author = (post.get("author") or {}).get("did")
        record = post.get("record") or {}
        text = record.get("text")
        if author == author_did and isinstance(text, str) and text.strip():
            out.append(text.strip())
    replies = node.get("replies")
    if isinstance(replies, list):
        for child in replies:
            _collect_thread_text(child, author_did, out)


@app.command()
def download(username: str = typer.Argument(..., help="Bluesky handle (e.g., user.bsky.social)"),
             limit: int = typer.Option(1000, help="Number of posts/threads to fetch (approximate)")) -> None:
    profile = resolve_profile(username)
    did = profile.get("did")
    if not did:
        typer.echo(f"Could not resolve DID for {username}")
        raise typer.Exit(1)
    target_dir = Path("skeets")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{username}.jsonl"
    seen_roots: set[str] = set()
    wrote = 0
    cursor: Optional[str] = None
    with out_path.open("w", encoding="utf-8") as f:
        while wrote < limit:
            data = get_author_feed(username, cursor=cursor, limit=100, filter_="posts_with_replies")
            items = data.get("feed") or []
            if not items:
                break
            cursor = data.get("cursor")
            for item in items:
                post = item.get("post") or {}
                if ((post.get("author") or {}).get("did")) != did:
                    continue
                uri = post.get("uri")
                if not uri:
                    continue
                record = post.get("record") or {}
                reply_info = record.get("reply") or {}
                root = ((reply_info.get("root") or {}).get("uri")) or uri
                if root in seen_roots:
                    continue
                seen_roots.add(root)
                try:
                    thread = get_post_thread(root, depth=100, parent_height=100)
                    node = thread.get("thread") or {}
                    texts: List[str] = []
                    _collect_thread_text(node, did, texts)
                    if not texts:
                        # Fallback to single post text
                        txt = (record.get("text") or "").strip()
                        if not txt:
                            continue
                        text = txt
                    else:
                        text = "\n".join(texts)
                    created = post.get("indexedAt") or post.get("createdAt") or ""
                    line = {"uri": root, "createdAt": created, "text": text}
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    wrote += 1
                    if wrote >= limit:
                        break
                except requests.HTTPError:
                    # Skip problematic threads; continue
                    continue
            if not cursor:
                break
            time.sleep(0.2)
    typer.echo(f"Saved {wrote} items to {out_path}")


if __name__ == "__main__":
    app()

