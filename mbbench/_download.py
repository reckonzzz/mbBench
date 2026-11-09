from __future__ import annotations

import hashlib
import stat
import sys
import time
import zipfile
from pathlib import Path
from typing import BinaryIO


def copy_download(
    response: BinaryIO,
    archive: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    progress: bool | None = None,
) -> None:
    """Write a response to disk and optionally verify and show its transfer."""

    stream = sys.stderr
    show = stream is not None and (
        getattr(stream, "isatty", lambda: False)() if progress is None else progress
    )
    length = str(getattr(response, "headers", {}).get("Content-Length", ""))
    total = int(length) if length.isascii() and length.isdigit() else 0
    digest = hashlib.sha256() if expected_sha256 is not None else None
    received = 0

    def report() -> None:
        size = f"{received / 1048576:.1f} MiB"
        if total > 0:
            fraction = min(received / total, 1.0)
            filled = int(fraction * 20)
            bar = "=" * filled + " " * (20 - filled)
            size = f"[{bar}] {fraction:4.0%}  {size} / {total / 1048576:.1f} MiB"
        print(f"\rDownloading {label}  {size}", file=stream, end="", flush=True)

    next_update = time.monotonic()
    if show:
        report()
    try:
        with archive.open("wb") as handle:
            for block in iter(lambda: response.read(256 * 1024), b""):
                handle.write(block)
                if digest is not None:
                    digest.update(block)
                received += len(block)
                if show and time.monotonic() >= next_update:
                    report()
                    next_update = time.monotonic() + 0.2
        if show:
            report()
    finally:
        if show:
            print(file=stream, flush=True)
    if digest is not None and digest.hexdigest() != expected_sha256:
        raise ValueError(f"{label} failed SHA-256 verification.")


def check_archive_path(directory: Path, name: str, *, label: str) -> None:
    resolved = (directory / name).resolve()
    if directory not in resolved.parents and resolved != directory:
        raise ValueError(f"{label} contains an unsafe path.")


def extract_zip(archive: Path, directory: Path, *, label: str) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for member in members:
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"{label} must not contain links.")
            check_archive_path(directory, member.filename, label=label)
        bundle.extractall(directory, members)
