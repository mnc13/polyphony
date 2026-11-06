"""
P10 — Human query gate (Step 10a).

After a case's central script is ready, prompt the user; a keypress of 'Y' within the
timeout holds the run for Socratic dialogue, otherwise the batch auto-advances. Uses a
timed read so an unattended batch is never blocked.

All stdin reads go through ONE persistent reader thread that feeds a queue. The gate
used to start a fresh thread per call, blocked in input(); a thread that timed out
stayed blocked there, so the next gate's "y" was consumed by the stale thread and that
gate timed out too (reproduced: "y" typed during gate 2 returned False). The reader
uses os.read on the raw descriptor, not the buffered sys.stdin: a daemon thread parked
in sys.stdin's buffered read holds its lock, and interpreter shutdown then aborted with
"Fatal Python error: _enter_buffered_busy" (exit 139) on piped stdin.
"""
from __future__ import annotations

import os
import queue
import sys
import threading

from config.settings import get_settings


class _LineReader:
    """Lines from one file descriptor, read on a daemon thread; None marks EOF."""

    def __init__(self, fd: int, encoding: str | None = None):
        self.fd = fd
        self.encoding = encoding or getattr(sys.stdin, "encoding", None) or "utf-8"
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.eof = False
        threading.Thread(target=self._run, daemon=True, name="stdin-reader").start()

    def _run(self) -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(self.fd, 1024)
            except OSError:
                chunk = b""
            if not chunk:
                if buf:
                    self.lines.put(buf.decode(self.encoding, "replace"))
                self.lines.put(None)
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.lines.put(line.rstrip(b"\r").decode(self.encoding, "replace"))

    def drain(self) -> None:
        """Drop lines typed before the current prompt (they answered nothing)."""
        while True:
            try:
                if self.lines.get_nowait() is None:
                    self.eof = True
            except queue.Empty:
                return

    def get(self, timeout: float | None) -> str | None:
        """Next line, or None on timeout / EOF."""
        if self.eof:
            return None
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            self.eof = True
        return line


_reader: _LineReader | None = None


def _get_reader() -> _LineReader:
    global _reader
    if _reader is None:
        _reader = _LineReader(sys.stdin.fileno())
    return _reader


def read_line(prompt: str = "", timeout: float | None = None) -> str | None:
    """Print `prompt`, return the next line typed (None on timeout or EOF)."""
    print(prompt, end="", flush=True)
    return _get_reader().get(timeout)


def ask_gate(prompt: str = "Any queries on this case? [Y] ", timeout: int | None = None) -> bool:
    """Return True if the user pressed Y within the timeout; False on timeout/other."""
    timeout = timeout if timeout is not None else get_settings().hp.query_gate_timeout_s
    reader = _get_reader()
    reader.drain()
    print(prompt, end="", flush=True)
    line = reader.get(timeout)
    if line is None and not reader.eof:
        print(f"\n[gate] no response in {timeout}s — advancing.")
        return False
    return (line or "").strip().lower() == "y"
