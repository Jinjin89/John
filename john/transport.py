"""Headless NDJSON subprocess transport and bidirectional JSON-RPC routing."""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from john.types import TransportError

type Frame = dict[str, Any]
type RequestId = int | str
type RpcParams = dict[str, Any] | list[Any] | None
type FrameHandler = Callable[[Frame], Awaitable[None]]
type MalformedHandler = Callable[[str, Exception], Awaitable[None]]
type EofHandler = Callable[[int | None], Awaitable[None]]
type ReaperPipe = tuple[asyncio.StreamReader, asyncio.ReadTransport]

_LOG = logging.getLogger(__name__)
_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_QUEUE_FRAMES = 1024
_MAX_QUEUE_BYTES = 32 * 1024 * 1024
_READ_BYTES = 64 * 1024
_REDACTED = "<redacted>"
_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<key>(?<![\w-])["']?
       (?:[\w-]*(?:(?:api|access|private)[_-]?key|token|password|secret)|authorization)
       ["']?[ \t]*[=:][ \t]*)
    (?:"(?:[^"\\\r\n]|\\.)*(?:"|$)|'(?:[^'\\\r\n]|\\.)*(?:'|$)|
       (?:(?:bearer|basic)[ \t]+)?[^\s,;"'\]}]+)
    """
)
_SECRET_TOKEN = re.compile(
    r"(?i)(?P<scheme>\b(?:bearer|basic)[ \t]+)[a-z0-9._~+/=-]+|\bsk-[a-z0-9_-]+"
)


def redact_text(text: str) -> str:
    """Mask common credentials in diagnostic text before it leaves the transport."""
    text = _SECRET_ASSIGNMENT.sub(r"\g<key>" + _REDACTED, text)
    return _SECRET_TOKEN.sub(lambda match: (match["scheme"] or "") + _REDACTED, text)


class StderrBuffer:
    """Keep a redacted tail and count raw line and byte totals, including delimiters."""

    def __init__(self, max_lines: int = 200, max_bytes: int = 65536) -> None:
        if max_lines < 0 or max_bytes < 0:
            raise ValueError("Stderr buffer limits must be nonnegative")
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.total_lines = 0
        self.total_bytes = 0
        self.truncated = False
        self._lines: deque[tuple[str, int]] = deque()
        self._retained_bytes = 0

    def append(self, line: str, *, raw_bytes: int | None = None) -> None:
        """Store one line. `raw_bytes` accounts for the original encoded input when known."""
        self.total_lines += 1
        self.total_bytes += len(line.encode("utf-8")) if raw_bytes is None else raw_bytes
        line = redact_text(line.removesuffix("\n").removesuffix("\r"))
        size = len(line.encode("utf-8")) + 1
        self._lines.append((line, size))
        self._retained_bytes += size
        while len(self._lines) > self.max_lines or self._retained_bytes > self.max_bytes:
            self._retained_bytes -= self._lines.popleft()[1]
            self.truncated = True

    def tail(self, n: int = 50) -> list[str]:
        return [line for line, _ in self._lines][-n:] if n > 0 else []

    def summary(self) -> str:
        status = " (tail truncated)" if self.truncated else ""
        return f"stderr: {self.total_lines} lines, {self.total_bytes} bytes{status}"


async def _lines(
    reader: asyncio.StreamReader, limit: int
) -> AsyncIterator[tuple[bytes | None, int]]:
    """Yield a bounded line and its wire size, or None for the body of an oversized line."""
    line = bytearray()
    size = 0
    while chunk := await reader.read(_READ_BYTES):
        *complete, remainder = chunk.split(b"\n")
        for part in complete:
            size += len(part)
            if size <= limit:
                line.extend(part)
                yield bytes(line), size + 1
            else:
                yield None, size + 1
            line.clear()
            size = 0
        size += len(remainder)
        if size <= limit:
            line.extend(remainder)
        else:
            line.clear()
    if size:
        yield (bytes(line) if size <= limit else None), size


# Run Linux's subreaper in a dedicated child so the embedding interpreter's child
# policy stays untouched. Only the private pipe carries lifecycle metadata.
_REAPER_BOOTSTRAP = """
import sys
sys.path.insert(0, sys.argv.pop(1))
from john.transport import _run_subreaper
_run_subreaper(int(sys.argv[1]), sys.argv[2:])
"""
_PR_SET_CHILD_SUBREAPER = 36


def _run_subreaper(status_fd: int, argv: Sequence[str]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Could not enable child reaping")
        worker = subprocess.Popen(argv, start_new_session=True)  # noqa: SIM115
    except OSError as exc:
        os.write(status_fd, f"error {exc.errno}\n".encode())
        os._exit(1)
    except ValueError:
        os.write(status_fd, b"error invalid arguments\n")
        os._exit(1)

    # The worker owns all three streams from here on, including their EOF semantics.
    for fd in (0, 1, 2):
        os.close(fd)
    os.write(status_fd, f"{worker.pid}\n".encode())
    exit_code = worker.wait()
    with suppress(ChildProcessError):
        while True:
            os.waitpid(-worker.pid, 0)
    os.write(status_fd, f"{exit_code}\n".encode())
    os._exit(0)


class _ProcessProtocol(asyncio.subprocess.SubprocessStreamProtocol):
    def __init__(self, limit: int, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(limit, loop)
        self.exited = asyncio.Event()

    def connection_made(self, transport: asyncio.SubprocessTransport) -> None:
        super().connection_made(transport)
        self.transport = transport

    def process_exited(self) -> None:
        super().process_exited()
        self.exited.set()


async def _spawn(
    argv: Sequence[str], cwd: Path | None, env: Mapping[str, str] | None, limit: int
) -> tuple[asyncio.subprocess.Process, _ProcessProtocol, int, ReaperPipe | None]:
    loop = asyncio.get_running_loop()

    async def spawn(
        command: Sequence[str], pass_fds: tuple[int, ...] = ()
    ) -> tuple[asyncio.subprocess.Process, _ProcessProtocol]:
        transport, protocol = await loop.subprocess_exec(
            lambda: _ProcessProtocol(limit, loop),
            *command,
            cwd=cwd,
            env=None if env is None else dict(env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        return asyncio.subprocess.Process(transport, protocol, loop), protocol

    try:
        if sys.platform != "linux":
            process, protocol = await spawn(argv)
            return process, protocol, process.pid, None

        read_fd, write_fd = os.pipe()
        status_file = os.fdopen(read_fd, "rb")
        status_reader = asyncio.StreamReader(limit=256)
        status_pipe = None
        try:
            try:
                status_pipe, _ = await loop.connect_read_pipe(
                    lambda: asyncio.StreamReaderProtocol(status_reader), status_file
                )
                command = (
                    sys.executable,
                    "-I",
                    "-c",
                    _REAPER_BOOTSTRAP,
                    str(Path(__file__).resolve().parents[1]),
                    str(write_fd),
                    *argv,
                )
                process, protocol = await spawn(command, (write_fd,))
            finally:
                os.close(write_fd)
            ready = await status_reader.readline()
            if ready.strip().isdigit():
                return process, protocol, int(ready), (status_reader, status_pipe)
            await _wait_after_status(process)
            await process.communicate()
            reason = ready.removeprefix(b"error ").strip().decode("ascii", errors="replace")
            raise TransportError(f"Subprocess startup failed ({reason or 'missing reaper status'})")
        except BaseException:
            if status_pipe is None:
                status_file.close()
            else:
                status_pipe.close()
            raise
    except (OSError, ValueError) as exc:
        raise TransportError(f"Subprocess startup failed ({type(exc).__name__})") from None


async def _wait_after_status(process: asyncio.subprocess.Process) -> None:
    # The reaper is exiting. Keep the loop ticking until asyncio's child watcher
    # publishes the exit status, including in sandboxes with restricted wakeups.
    while process.returncode is None:
        await asyncio.sleep(0.01)
    await process.wait()


def _reject_constant(value: str) -> None:
    raise ValueError("JSON numbers must be finite")


class FrameQueue[T]:
    """An ordered handoff that shares the transport's backlog budget."""

    def __init__(self, transport: StdioTransport) -> None:
        self._transport = transport
        self._queue: asyncio.Queue[tuple[T, int] | None] = asyncio.Queue()
        self._finished = False

    def put_nowait(self, item: T, size: int) -> bool:
        if self._finished or not self._transport._reserve_queue(size):
            return False
        self._queue.put_nowait((item, size))
        return True

    async def get(self) -> tuple[T, int] | None:
        entry = await self._queue.get()
        if entry is not None:
            self._transport._release_queue(entry[1])
        return entry

    def finish(self) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(None)

    def clear(self) -> None:
        self._finished = True
        while not self._queue.empty():
            entry = self._queue.get_nowait()
            if entry is not None:
                self._transport._release_queue(entry[1])


class StdioTransport:
    """Read subprocess pipes independently and dispatch stdout frames in wire order."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        protocol: _ProcessProtocol,
        pid: int,
        status: ReaperPipe | None,
        on_frame: FrameHandler,
        on_malformed: MalformedHandler | None,
        on_eof: EofHandler | None,
        max_line_bytes: int,
        max_queue_frames: int,
        max_queue_bytes: int,
    ) -> None:
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        self._pid = pid
        self.stderr = StderrBuffer()
        self.error: TransportError | None = None
        self._process = process
        self._protocol = protocol
        self._status = status
        self._stdin = process.stdin
        self._write_lock = asyncio.Lock()
        self._inbox: FrameQueue[Frame | tuple[str, TransportError]] = FrameQueue(self)
        self._max_queue_frames = max_queue_frames
        self._max_queue_bytes = max_queue_bytes
        self._queued_frames = 0
        self._queued_bytes = 0
        self._dispatch_frame_size = 0
        self._ended = False
        self._end_callbacks: list[Callable[[bool], None]] = []
        self._handlers: set[asyncio.Task[None]] = set()
        self._reap_task: asyncio.Task[int] | None = None
        self._close_task: asyncio.Task[int] | None = None
        self._exit_task = asyncio.create_task(self._wait_for_process(), name="stdio process exit")
        self._stdout_task = asyncio.create_task(
            self._read_stdout_loop(process.stdout, max_line_bytes), name="stdio stdout"
        )
        self._stderr_task = asyncio.create_task(
            self._read_stderr_loop(process.stderr), name="stdio stderr"
        )
        self._dispatch_task = asyncio.create_task(
            self._dispatch(on_frame, on_malformed, on_eof), name="stdio frame dispatch"
        )

    @classmethod
    async def start(
        cls,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        on_frame: FrameHandler,
        on_malformed: MalformedHandler | None = None,
        on_eof: EofHandler | None = None,
        max_line_bytes: int = _MAX_LINE_BYTES,
        max_queue_frames: int = _MAX_QUEUE_FRAMES,
        max_queue_bytes: int = _MAX_QUEUE_BYTES,
    ) -> StdioTransport:
        """Spawn explicit argv with a shared cap on queued frames and their wire bytes.

        An explicit env replaces the inherited environment.
        """
        if not argv or isinstance(argv, (str, bytes)) or max_line_bytes <= 0:
            raise TransportError("A nonempty argv sequence and positive line limit are required")
        if os.name != "posix":
            raise TransportError("The subprocess transport requires POSIX process groups")
        if max_queue_frames <= 0 or max_queue_bytes <= 0:
            raise TransportError("Transport frame queue limits must be positive")

        startup = asyncio.create_task(_spawn(argv, cwd, env, max_line_bytes), name="stdio startup")
        limits = max_line_bytes, max_queue_frames, max_queue_bytes
        try:
            spawned = await asyncio.shield(startup)
        except asyncio.CancelledError:

            async def abandon() -> None:
                async def discard(*args: object) -> None:
                    pass

                with suppress(TransportError):
                    transport = cls(*(await startup), discard, discard, None, *limits)
                    await transport.close(grace_s=0, term_s=0)

            await asyncio.shield(asyncio.create_task(abandon(), name="stdio abandoned startup"))
            raise
        return cls(*spawned, on_frame, on_malformed, on_eof, *limits)

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def closed(self) -> bool:
        return self._ended or self._close_task is not None

    @property
    def returncode(self) -> int | None:
        if (
            self._exit_task.done()
            and not self._exit_task.cancelled()
            and self._exit_task.exception() is None
        ):
            return self._exit_task.result()
        return None

    def _add_end_callback(self, callback: Callable[[bool], None]) -> None:
        if self._close_task is None or not self._close_task.done():
            self._end_callbacks.append(callback)
        if self._close_task is not None:
            callback(True)
        if self._ended:
            callback(False)

    def _track_handler(self, task: asyncio.Task[None]) -> None:
        self._handlers.add(task)
        task.add_done_callback(self._handlers.discard)

    def _fail(self, error: TransportError) -> None:
        if self.error is None:
            self.error = error
            _LOG.error("%s", redact_text(str(error)))
        self._begin_close(0, 0.2)

    def _has_queue_room(self, size: int) -> bool:
        return (
            self._queued_frames < self._max_queue_frames
            and self._queued_bytes + size <= self._max_queue_bytes
        )

    def _reserve_queue(self, size: int) -> bool:
        if self.error is not None:
            return False
        if not self._has_queue_room(size):
            self._fail(TransportError("Transport frame queue overflow"))
            return False
        self._queued_frames += 1
        self._queued_bytes += size
        return True

    def _release_queue(self, size: int) -> None:
        self._queued_frames -= 1
        self._queued_bytes -= size

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise TransportError(str(self.error))

    def frame_size(self, frame: Frame) -> int:
        """Use the original wire size during dispatch and estimate it for other callers."""
        if asyncio.current_task() is self._dispatch_task and self._dispatch_frame_size:
            return self._dispatch_frame_size
        return len(json.dumps(frame, separators=(",", ":"))) + 1

    async def _read_stdout_loop(self, reader: asyncio.StreamReader, limit: int) -> None:
        try:
            async for line, size in _lines(reader, limit):
                item: Frame | tuple[str, TransportError]
                if line is None:
                    error = TransportError(f"NDJSON line exceeds the {limit}-byte limit")
                    item = "<oversized line>", error
                elif not line.strip():
                    continue
                else:
                    try:
                        item = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
                        if not isinstance(item, dict):
                            raise ValueError("NDJSON frames must be objects")
                    except (ValueError, RecursionError) as exc:
                        preview = redact_text(line.decode("utf-8", errors="replace"))[:512]
                        item = preview, TransportError(f"Malformed NDJSON frame: {exc}")
                if isinstance(item, tuple):
                    size = len(item[0].encode("utf-8")) + len(str(item[1]).encode("utf-8"))
                if not self._has_queue_room(size):
                    # Give ready consumers a chance to free space in a burst of input.
                    await asyncio.sleep(0)
                if not self._inbox.put_nowait(item, size):
                    return
        except Exception as exc:
            self._fail(TransportError(f"Stdout reader failed ({type(exc).__name__})"))
        finally:
            self._inbox.finish()

    async def _read_stderr_loop(self, reader: asyncio.StreamReader) -> None:
        try:
            async for line, size in _lines(reader, self.stderr.max_bytes):
                if line is None:
                    self.stderr.truncated = True
                    self.stderr.append("<oversized stderr line>", raw_bytes=size)
                else:
                    self.stderr.append(line.decode("utf-8", errors="replace"), raw_bytes=size)
        except Exception as exc:
            self._fail(TransportError(f"Stderr reader failed ({type(exc).__name__})"))

    async def _dispatch(
        self,
        on_frame: FrameHandler,
        on_malformed: MalformedHandler | None,
        on_eof: EofHandler | None,
    ) -> None:
        try:
            while (entry := await self._inbox.get()) is not None and self.error is None:
                item, size = entry
                if isinstance(item, tuple):
                    preview, error = item
                    if on_malformed is None:
                        self._fail(error)
                        return
                    await on_malformed(preview, error)
                else:
                    self._dispatch_frame_size = size
                    try:
                        await on_frame(item)
                    finally:
                        self._dispatch_frame_size = 0
        except Exception as exc:
            self._fail(TransportError(f"Frame dispatch failed ({type(exc).__name__})"))
        finally:
            self._inbox.clear()
            self._ended = True
            for callback in self._end_callbacks:
                callback(False)
            if on_eof is not None:
                try:
                    await on_eof(self.returncode)
                except Exception as exc:
                    self._fail(TransportError(f"EOF callback failed ({type(exc).__name__})"))

    async def send_json(self, payload: Frame) -> None:
        async with self._write_lock:
            if self.closed or self._stdin.is_closing():
                raise TransportError(str(self.error or "Subprocess transport is closed"))
            try:
                if not isinstance(payload, dict):
                    raise TypeError("NDJSON frames must be objects")
                line = json.dumps(
                    payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                data = line.encode("utf-8") + b"\n"
            except (ValueError, TypeError, RecursionError):
                raise TransportError("Could not encode an NDJSON object") from None
            try:
                self._stdin.write(data)
                await self._stdin.drain()
            except (OSError, RuntimeError):
                error = TransportError(str(self.error or "Subprocess stdin is unavailable"))
                if not self.closed:
                    self._fail(error)
                raise error from None

    async def _wait_for_process(self) -> int:
        if self._status is None:
            # Process.wait() can wait on pipe EOF if called before child exit.
            await self._protocol.exited.wait()
            return await self._process.wait()
        reader, pipe = self._status
        try:
            line = await reader.read()
            await _wait_after_status(self._process)
            try:
                return int(line)
            except ValueError:
                raise TransportError("Subprocess reaper exited without reporting status") from None
        finally:
            pipe.close()

    def _signal_group(self, sig: int) -> bool:
        try:
            os.killpg(self.pid, sig)
        except ProcessLookupError:
            return False
        return True

    async def _wait_for_group(self) -> int:
        code = await asyncio.shield(self._exit_task)
        if self._status is None:
            # On other POSIX systems the system reaper collects adopted descendants.
            while self._signal_group(0):
                await asyncio.sleep(0.01)
        return code

    async def _reap(self, grace_s: float, term_s: float) -> int:
        self._stdin.close()
        for delay, sig in ((grace_s, signal.SIGTERM), (term_s, signal.SIGKILL)):
            try:
                async with asyncio.timeout(delay):
                    return await self._wait_for_group()
            except TimeoutError:
                if not self._exit_task.done() or self._status is None:
                    self._signal_group(sig)
        return await self._wait_for_group()

    async def _complete_close(self, reaping: asyncio.Task[int], drain_s: float) -> int:
        try:
            return await reaping
        finally:
            stdin = asyncio.create_task(self._stdin.wait_closed(), name="stdio stdin close")
            readers = self._stdout_task, self._stderr_task
            _, draining = await asyncio.wait((stdin, *readers), timeout=drain_s)
            for fd, task in enumerate(readers, start=1):
                if task in draining:
                    task.cancel()
                if pipe := self._protocol.transport.get_pipe_transport(fd):
                    pipe.close()
            if stdin in draining:
                self._stdin.transport.abort()
            await asyncio.gather(stdin, *readers, return_exceptions=True)
            _, pending = await asyncio.wait({self._dispatch_task, *self._handlers}, timeout=drain_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._end_callbacks.clear()

    def _begin_close(self, grace_s: float, term_s: float) -> asyncio.Task[int]:
        if self._close_task is None:
            self._reap_task = asyncio.create_task(
                self._reap(grace_s, term_s), name="stdio group reaping"
            )
            self._close_task = asyncio.create_task(
                self._complete_close(self._reap_task, grace_s), name="stdio close"
            )
            self._close_task.add_done_callback(_report_close_failure)
            for callback in self._end_callbacks:
                callback(True)
        return self._close_task

    async def close(self, *, grace_s: float = 2.0, term_s: float = 2.0) -> int | None:
        """Close stdin and join one shared, cancellation-shielded shutdown."""
        if any(not math.isfinite(delay) or delay < 0 for delay in (grace_s, term_s)):
            raise ValueError("Shutdown timeouts must be finite and nonnegative")
        completion = self._begin_close(grace_s, term_s)
        caller = asyncio.current_task()
        if caller is self._dispatch_task or caller in self._handlers:
            assert self._reap_task is not None
            completion = self._reap_task
        return await asyncio.shield(completion)

    async def __aenter__(self) -> StdioTransport:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def _report_close_failure(task: asyncio.Task[int]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        _LOG.error("Process cleanup failed (%s)", type(error).__name__)


class JsonRpcError(Exception):
    """An error response from a peer, with its code and structured data."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.message = redact_text(message)
        self.data = data
        super().__init__(self.message)


class MethodNotFound(JsonRpcError):
    def __init__(self, message: str = "Method not found") -> None:
        super().__init__(-32601, message)


type NotificationHandler = Callable[[str, RpcParams], Awaitable[None]]
type RequestHandler = Callable[[str, RpcParams, RequestId], Awaitable[Any]]


def _is_request_id(value: Any) -> bool:
    # bool is an int subclass, and Python considers 1, 1.0 and True equal.
    return type(value) in (int, str)


class JsonRpcPeer:
    """Route responses immediately, ordered notifications serially, and reverse calls concurrently.

    Wire `handle_frame` into an existing transport's `on_frame`, or use `start`
    to bind the two before dispatch. A request handler returns the response value
    or raises `JsonRpcError` to select an error reply. An unknown-frame handler
    explicitly opts in to raw frames.
    """

    def __init__(
        self,
        transport: StdioTransport,
        *,
        include_jsonrpc: bool = True,
        on_notification: NotificationHandler | None = None,
        on_request: RequestHandler | None = None,
        on_unknown_frame: FrameHandler | None = None,
    ) -> None:
        self.transport = transport
        self._include_jsonrpc = include_jsonrpc
        self._on_notification = on_notification
        self._on_request = on_request
        self._on_unknown_frame = on_unknown_frame
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._requests: set[asyncio.Task[None]] = set()
        self._notifications: FrameQueue[tuple[bool, Frame]] = FrameQueue(transport)
        self._delivery: asyncio.Task[None] | None = None
        self._disconnected = False
        if on_notification is not None or on_unknown_frame is not None:
            self._delivery = asyncio.create_task(self._deliver(), name="JSON-RPC notifications")
            self._delivery.add_done_callback(self._callback_done)
            transport._track_handler(self._delivery)
        transport._add_end_callback(self._disconnect)

    @classmethod
    async def start(
        cls,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        include_jsonrpc: bool = True,
        on_notification: NotificationHandler | None = None,
        on_request: RequestHandler | None = None,
        on_unknown_frame: FrameHandler | None = None,
        on_malformed: MalformedHandler | None = None,
        on_eof: EofHandler | None = None,
        max_line_bytes: int = _MAX_LINE_BYTES,
        max_queue_frames: int = _MAX_QUEUE_FRAMES,
        max_queue_bytes: int = _MAX_QUEUE_BYTES,
    ) -> JsonRpcPeer:
        attached: asyncio.Future[JsonRpcPeer] = asyncio.get_running_loop().create_future()

        async def route(frame: Frame) -> None:
            await (await attached).handle_frame(frame)

        transport = await StdioTransport.start(
            argv,
            cwd=cwd,
            env=env,
            on_frame=route,
            on_malformed=on_malformed,
            on_eof=on_eof,
            max_line_bytes=max_line_bytes,
            max_queue_frames=max_queue_frames,
            max_queue_bytes=max_queue_bytes,
        )
        peer = cls(
            transport,
            include_jsonrpc=include_jsonrpc,
            on_notification=on_notification,
            on_request=on_request,
            on_unknown_frame=on_unknown_frame,
        )
        attached.set_result(peer)
        return peer

    async def _send(self, **fields: Any) -> None:
        await self.transport.send_json(
            {"jsonrpc": "2.0", **fields} if self._include_jsonrpc else fields
        )

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        if self._disconnected:
            raise TransportError(str(self.transport.error or "JSON-RPC peer is closed"))
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        fields = {"params": params} if params is not None else {}
        try:
            async with asyncio.timeout(timeout_s):
                await self._send(id=request_id, method=method, **fields)
                return await future
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if self._disconnected and caller is not None and not caller.cancelling():
                error = self.transport.error or "JSON-RPC peer closed with a request pending"
                raise TransportError(str(error)) from None
            raise
        finally:
            self._pending.pop(request_id, None)
            if future.done() and not future.cancelled():
                future.exception()
            else:
                future.cancel()

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        fields = {"params": params} if params is not None else {}
        await self._send(method=method, **fields)

    async def reply_result(self, req_id: RequestId, result: Any) -> None:
        await self._send(id=req_id, result=result)

    async def reply_error(
        self, req_id: RequestId, code: int, message: str, data: Any = None
    ) -> None:
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        await self._send(id=req_id, error=error)

    async def handle_frame(self, frame: Frame) -> None:
        """Route a wire frame without awaiting a driver callback."""
        req_id = frame.get("id")
        method = frame.get("method")
        params = frame.get("params")
        if frame.get("jsonrpc", "2.0") == "2.0":
            if isinstance(method, str) and "result" not in frame and "error" not in frame:
                if params is None or isinstance(params, (dict, list)):
                    if "id" not in frame:
                        if self._on_notification is not None:
                            self._queue_notification(frame, unknown=False)
                        return
                    if _is_request_id(req_id):
                        if not self._disconnected:
                            task = asyncio.create_task(
                                self._serve_request(method, params, req_id),
                                name="JSON-RPC reverse request",
                            )
                            self._requests.add(task)
                            task.add_done_callback(self._callback_done)
                            self.transport._track_handler(task)
                        return
            elif "method" not in frame and type(req_id) is int:
                future = self._pending.get(req_id)
                if future is not None and not future.done() and self._resolve(future, frame):
                    return
        if self._on_unknown_frame is not None:
            self._queue_notification(frame, unknown=True)

    def _queue_notification(self, frame: Frame, *, unknown: bool) -> None:
        if self._delivery is not None and not self._delivery.done():
            self._notifications.put_nowait((unknown, frame), self.transport.frame_size(frame))

    @staticmethod
    def _resolve(future: asyncio.Future[Any], frame: Frame) -> bool:
        if "result" in frame and "error" not in frame:
            future.set_result(frame["result"])
            return True
        if "error" in frame and "result" not in frame:
            error = frame["error"]
            if (
                isinstance(error, dict)
                and type(error.get("code")) is int
                and isinstance(error.get("message"), str)
            ):
                future.set_exception(
                    JsonRpcError(error["code"], error["message"], error.get("data"))
                )
                return True
        future.set_exception(TransportError("Malformed JSON-RPC response"))
        return False

    async def _serve_request(self, method: str, params: RpcParams, req_id: RequestId) -> None:
        try:
            try:
                if self._on_request is None:
                    raise MethodNotFound
                result = await self._on_request(method, params, req_id)
            except JsonRpcError as exc:
                await self.reply_error(req_id, exc.code, exc.message, exc.data)
            except Exception:
                await self.reply_error(req_id, -32603, "Internal error")
            else:
                await self.reply_result(req_id, result)
        except TransportError:
            if not self.transport.closed:
                await self.reply_error(req_id, -32603, "Invalid response from request handler")

    async def _deliver(self) -> None:
        try:
            while (
                entry := await self._notifications.get()
            ) is not None and self.transport.error is None:
                (unknown, frame), _ = entry
                if unknown:
                    assert self._on_unknown_frame is not None
                    await self._on_unknown_frame(frame)
                else:
                    assert self._on_notification is not None
                    await self._on_notification(frame["method"], frame.get("params"))
        finally:
            self._notifications.clear()

    def _callback_done(self, task: asyncio.Task[None]) -> None:
        self._requests.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            self.transport._fail(
                TransportError(f"JSON-RPC dispatch failed ({type(error).__name__})")
            )

    def _disconnect(self, closing: bool) -> None:
        current = asyncio.current_task()
        if not self._disconnected:
            self._disconnected = True
            for future in self._pending.values():
                future.cancel()
            self._pending.clear()
            for task in self._requests:
                if task is not current:
                    task.cancel()
        if not closing and self._delivery is not None and not self._delivery.done():
            self._notifications.finish()

    async def close(self, *, grace_s: float = 2.0, term_s: float = 2.0) -> int | None:
        return await self.transport.close(grace_s=grace_s, term_s=term_s)
