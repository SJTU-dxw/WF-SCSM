#!/usr/bin/env python3
"""Resumable multipart upload to a Zenodo draft record."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

SESSION = requests.Session()
SESSION.trust_env = False


class FileSlice:
    def __init__(self, path: Path, offset: int, length: int):
        self._file = path.open("rb")
        self._file.seek(offset)
        self._remaining = length

    def __len__(self) -> int:
        return self._remaining

    def read(self, size: int = -1) -> bytes:
        if self._remaining == 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._file.read(size)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        self._file.close()


def request_with_retry(method, url, *, attempts=12, **kwargs):
    for attempt in range(1, attempts + 1):
        try:
            response = SESSION.request(method, url, **kwargs)
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response
            detail = f"HTTP {response.status_code}: {response.text[:300]}"
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            detail = str(exc)
        if attempt == attempts:
            raise RuntimeError(f"request failed after {attempts} attempts: {detail}")
        delay = min(60, 2 ** min(attempt, 5))
        print(f"  retry {attempt}/{attempts} in {delay}s ({detail})", flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("files_url", help="draft links.files URL")
    parser.add_argument("--part-mib", type=int, default=5)
    args = parser.parse_args()

    source = args.file.resolve()
    if not source.is_file():
        parser.error(f"file not found: {source}")
    if args.part_mib < 5:
        parser.error("--part-mib must be at least 5")

    token = os.environ.get("ZENODO_PAT") or getpass.getpass("Zenodo PAT: ")
    headers = {"Authorization": f"Bearer {token}"}
    files_url = args.files_url.rstrip("/")
    size = source.stat().st_size
    part_size = args.part_mib * 1024 * 1024
    part_count = math.ceil(size / part_size)
    state_path = source.with_name(f".{source.name}.zenodo-multipart.json")

    if state_path.exists():
        state = json.loads(state_path.read_text())
        expected = (str(source), size, files_url, part_size)
        actual = (state["file"], state["size"], state["files_url"], state["part_size"])
        if actual != expected:
            raise RuntimeError(f"state file does not match this upload: {state_path}")
        print(f"Resuming multipart upload ({len(state['completed_parts'])}/{state['part_count']} parts complete)", flush=True)
    else:
        metadata = [{
            "key": source.name,
            "size": size,
            "transfer": {"type": "M", "parts": part_count, "part_size": part_size},
        }]
        response = request_with_retry("POST", files_url, json=metadata, headers=headers, timeout=60)
        entry = next(item for item in response.json()["entries"] if item["key"] == source.name)
        part_links = sorted(entry["links"]["parts"], key=lambda item: item["part"])
        if len(part_links) != part_count:
            raise RuntimeError(f"server returned {len(part_links)} part URLs, expected {part_count}")
        state = {
            "file": str(source),
            "size": size,
            "files_url": files_url,
            "part_size": part_size,
            "part_count": part_count,
            "commit_url": entry["links"]["commit"],
            "part_links": part_links,
            "completed_parts": [],
        }
        save_state(state_path, state)
        print(f"Started multipart upload: {part_count} parts", flush=True)

    completed = set(state["completed_parts"])
    started = time.monotonic()
    for part_number in range(part_count):
        if part_number in completed:
            continue
        offset = part_number * part_size
        length = min(part_size, size - offset)
        part_url = state["part_links"][part_number]["url"]
        part_headers = None if "X-Amz-Signature" in part_url or "Signature=" in part_url else headers
        with source.open("rb") as source_file:
            source_file.seek(offset)
            chunk = source_file.read(length)
            for attempt in range(1, 13):
                try:
                    response = SESSION.put(part_url, data=chunk, headers=part_headers, timeout=600)
                    if response.status_code < 500 and response.status_code != 429:
                        response.raise_for_status()
                        break
                    detail = f"HTTP {response.status_code}: {response.text[:300]}"
                except requests.HTTPError:
                    raise
                except requests.RequestException as exc:
                    detail = str(exc)
                if attempt == 12:
                    raise RuntimeError(f"part {part_number} failed: {detail}")
                delay = min(60, 2 ** min(attempt, 5))
                print(f"  part {part_number} retry {attempt}/12 in {delay}s ({detail})", flush=True)
                time.sleep(delay)
        completed.add(part_number)
        state["completed_parts"] = sorted(completed)
        save_state(state_path, state)
        percent = 100 * min((part_number + 1) * part_size, size) / size
        elapsed = time.monotonic() - started
        print(f"part {part_number + 1}/{part_count} complete ({percent:.1f}%, {elapsed / 60:.1f} min)", flush=True)

    response = request_with_retry("POST", state["commit_url"], json={}, headers=headers, timeout=1800)
    if response.content:
        print(json.dumps(response.json(), indent=2), flush=True)
    state_path.unlink(missing_ok=True)
    print("Upload complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; run the same command to resume.", file=sys.stderr)
        sys.exit(130)
