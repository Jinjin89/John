import asyncio
import os
import signal
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from john.transport import (
    JsonRpcError,
    JsonRpcPeer,
    MethodNotFound,
    StderrBuffer,
    StdioTransport,
    redact_text,
)
from john.types import TransportError


def child(source: str, *args: str) -> tuple[str, ...]:
    return (sys.executable, "-u", "-c", dedent(source), *args)


async def ignore_frame(frame: dict[str, Any]) -> None:
    pass


def test_redaction_and_byte_bounded_ring() -> None:
    text = (
        'ANTHROPIC_API_KEY="secret with spaces" OPENAI_API_KEY=plain-secret '
        "Authorization: Bearer header-secret standalone Bearer other-secret "
        'token=url-secret "api_key": "json-secret" raw sk-secret123'
    )
    masked = redact_text(text)
    assert "secret" not in masked
    assert masked.count("<redacted>") == 7
    assert redact_text(masked) == masked

    buffer = StderrBuffer(max_lines=3, max_bytes=7)
    for line in ("é\n", "é\n", "é\n", "z\n"):
        buffer.append(line)
    assert buffer.total_lines == 4
    assert buffer.total_bytes == 11
    assert buffer.tail() == ["é", "z"]
    assert buffer.tail(1) == ["z"]
    assert buffer.tail(0) == []
    assert buffer.truncated and "truncated" in buffer.summary()


async def test_large_frames_and_concurrent_writes() -> None:
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    transport = await StdioTransport.start(
        child("""
            import json
            import sys
            for line in sys.stdin:
                print(json.dumps(json.loads(line), ensure_ascii=False), flush=True)
        """),
        on_frame=frames.put,
    )
    content = "abcde世界\n" * 30_000
    assert len(content.encode()) > 256 * 1024
    async with transport, asyncio.timeout(8):
        await asyncio.gather(
            *(transport.send_json({"id": index, "text": content}) for index in range(4))
        )
        received = [await frames.get() for _ in range(4)]
        assert sorted(frame["id"] for frame in received) == [0, 1, 2, 3]
        assert all(frame["text"] == content for frame in received)
    assert transport.returncode == 0


async def test_stderr_flood_remains_bounded_and_redacted() -> None:
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    transport = await StdioTransport.start(
        child(r"""
            import json
            import sys
            line = b'token=stderr-token Authorization: Bearer header-token sk-secret123 '
            line += b'x' * 100 + b'\x0a'
            for _ in range(5000):
                sys.stderr.buffer.write(line)
            oversized = b'y' * 300000 + b' token=oversized-secret\x0a'
            sys.stderr.buffer.write(oversized)
            sys.stderr.buffer.write(b'incomplete final line')
            sys.stderr.flush()
            print(json.dumps({'bytes': len(line) * 5000 + len(oversized) + 21}), flush=True)
        """),
        on_frame=frames.put,
    )
    async with transport, asyncio.timeout(8):
        frame = await frames.get()
        assert frame["bytes"] > 512 * 1024
    buffer = transport.stderr
    assert buffer.total_lines == 5002
    assert buffer.total_bytes == frame["bytes"]
    assert buffer.truncated
    tail = buffer.tail(5000)
    assert len(tail) == 200
    assert sum(len(line.encode()) + 1 for line in tail) <= 65536
    assert tail[-2:] == ["<oversized stderr line>", "incomplete final line"]
    assert "<redacted>" in tail[0]
    assert all(
        secret not in "\n".join(tail)
        for secret in (
            "stderr-token",
            "header-token",
            "sk-secret123",
            "oversized-secret",
        )
    )


async def test_stdout_keeps_draining_while_callback_waits(tmp_path: Path) -> None:
    written = tmp_path / "written"
    first = asyncio.Event()
    release = asyncio.Event()
    frames: asyncio.Queue[int] = asyncio.Queue()

    async def on_frame(frame: dict[str, Any]) -> None:
        if frame["seq"] == 0:
            first.set()
            await release.wait()
        frames.put_nowait(frame["seq"])

    transport = await StdioTransport.start(
        child(
            """
            import json
            import sys
            from pathlib import Path
            for seq in range(4):
                print(json.dumps({'seq': seq, 'body': 'x' * 300000}), flush=True)
            Path(sys.argv[1]).touch()
        """,
            str(written),
        ),
        on_frame=on_frame,
    )
    async with transport, asyncio.timeout(8):
        await first.wait()
        while not written.exists():
            await asyncio.sleep(0.01)
        assert frames.empty()
        release.set()
        assert [await frames.get() for _ in range(4)] == [0, 1, 2, 3]


@pytest.mark.parametrize("max_frames,max_bytes", [(2, 4096), (64, 200)], ids=["frames", "bytes"])
async def test_stdout_queue_overflow_stops_a_slow_consumer(max_frames: int, max_bytes: int) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def on_frame(frame: dict[str, Any]) -> None:
        assert frame == {"ready": True}
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    transport = await StdioTransport.start(
        child("""
            import json, sys
            print('{"ready":true}')
            input()
            for seq in range(12):
                print(json.dumps({'seq': seq, 'body': 'x' * 80}))
            sys.stdin.read()
        """),
        on_frame=on_frame,
        max_queue_frames=max_frames,
        max_queue_bytes=max_bytes,
    )
    async with transport, asyncio.timeout(3):
        await started.wait()
        await transport.send_json({"flood": True})
        await stopped.wait()
        assert transport.closed and isinstance(transport.error, TransportError)
        assert str(transport.error) == "Transport frame queue overflow"
        with pytest.raises(TransportError, match="Transport frame queue overflow"):
            await transport.send_json({"late": True})


async def test_malformed_and_oversized_stdout_resynchronize() -> None:
    events: asyncio.Queue[dict[str, Any] | tuple[str, Exception]] = asyncio.Queue()

    async def malformed(line: str, error: Exception) -> None:
        events.put_nowait((line, error))

    transport = await StdioTransport.start(
        child(r"""
            import sys
            print('broken token=diagnostic-secret', flush=True)
            sys.stdout.buffer.write(b'x' * 100000 + b'{"false_start":true}\n')
            sys.stdout.buffer.write(b'{"ok":true}\r\n{"last":true}')
            sys.stdout.flush()
        """),
        max_line_bytes=64,
        on_frame=events.put,
        on_malformed=malformed,
    )
    async with transport, asyncio.timeout(8):
        preview, error = await events.get()
        assert (preview, type(error)) == ("broken token=<redacted>", TransportError)
        preview, error = await events.get()
        assert preview == "<oversized line>" and "64-byte limit" in str(error)
        assert await events.get() == {"ok": True}
        assert await events.get() == {"last": True}
    assert transport.error is None


async def test_unhandled_malformed_stdout_fails_and_closes() -> None:
    ended = asyncio.Event()

    async def eof(code: int | None) -> None:
        ended.set()

    transport = await StdioTransport.start(
        child("""
            import signal
            print('invalid', flush=True)
            signal.pause()
        """),
        on_frame=ignore_frame,
        on_eof=eof,
    )
    async with asyncio.timeout(8):
        await ended.wait()
        assert await transport.close() == -signal.SIGTERM
        assert isinstance(transport.error, TransportError)
        assert "Malformed NDJSON frame" in str(transport.error)
        with pytest.raises(TransportError, match="Malformed NDJSON frame"):
            await transport.send_json({"late": True})


async def test_spawn_scope_and_launch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOHN_TRANSPORT_PARENT_ONLY", "ambient")
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    transport = await StdioTransport.start(
        child("""
            import json
            import os
            print(json.dumps({'cwd': os.getcwd(), 'flag': os.getenv('JOHN_TRANSPORT_LOCAL'),
                              'parent': os.getenv('JOHN_TRANSPORT_PARENT_ONLY')}), flush=True)
        """),
        cwd=tmp_path,
        env={"JOHN_TRANSPORT_LOCAL": "sent"},
        on_frame=frames.put,
    )
    async with transport, asyncio.timeout(8):
        assert await frames.get() == {"cwd": str(tmp_path), "flag": "sent", "parent": None}
    with pytest.raises(TransportError, match="startup failed"):
        await StdioTransport.start((str(tmp_path / "absent"),), on_frame=ignore_frame)


async def test_initial_callback_can_send_with_its_transport() -> None:
    done = asyncio.Event()

    async def on_frame(frame: dict[str, Any]) -> None:
        if frame["ready"]:
            await transport.send_json({"acknowledge": True})
        else:
            done.set()

    transport = await StdioTransport.start(
        child("""
            import json
            print('{"ready":true}', flush=True)
            assert json.loads(input()) == {'acknowledge': True}
            print('{"ready":false}', flush=True)
        """),
        on_frame=on_frame,
    )
    async with transport, asyncio.timeout(8):
        await done.wait()
    assert transport.returncode == 0


@pytest.mark.parametrize("include_jsonrpc", [True, False])
async def test_rpc_duplex_with_blocked_approval_and_ordered_notifications(
    include_jsonrpc: bool,
) -> None:
    notices: list[str] = []
    unknown: list[dict[str, Any]] = []
    answers: list[tuple[str, object, int | str]] = []
    approval_started = asyncio.Event()
    release_approval = asyncio.Event()
    notice_started = asyncio.Event()
    release_notice = asyncio.Event()
    checked = asyncio.Event()

    async def on_notification(method: str, params: object) -> None:
        if method == "first":
            notice_started.set()
            await release_notice.wait()
        notices.append(method)
        if method == "checked":
            checked.set()

    async def on_request(method: str, params: object, req_id: int | str) -> dict[str, bool]:
        answers.append((method, params, req_id))
        if method != "approve":
            raise MethodNotFound
        approval_started.set()
        await release_approval.wait()
        return {"allowed": False}

    async def on_unknown(frame: dict[str, Any]) -> None:
        unknown.append(frame)

    peer = await JsonRpcPeer.start(
        child(
            """
            import json
            import sys
            header = {'jsonrpc': '2.0'} if sys.argv[1] == 'versioned' else {}

            def send(**fields):
                print(json.dumps({**header, **fields}), flush=True)

            def receive():
                message = json.loads(input())
                assert ('jsonrpc' in message) == bool(header)
                return message

            request = receive()
            assert request['method'] == 'turn' and request['id'] == 0
            send(method='first')
            send(method='approve', id='0', params={'name': 'tool'})
            send(method='unknown', id=0)
            send(id='0', result='wrong string ID')
            send(id=False, result='wrong boolean ID')
            send(method='second')
            unknown = receive()
            assert type(unknown['id']) is int and unknown['id'] == 0
            assert unknown['error']['code'] == -32601
            send(id=request['id'], result='turn reply')
            approval = receive()
            assert approval['id'] == '0' and approval['result'] == {'allowed': False}
            send(method='checked')
            notice = receive()
            assert notice['method'] == 'done' and 'id' not in notice
            request = receive()
            assert request['id'] == 1
            error = {'code': -32000, 'message': 'Call failed', 'data': {'detail': 7}}
            send(id=request['id'], error=error)
        """,
            "versioned" if include_jsonrpc else "unversioned",
        ),
        include_jsonrpc=include_jsonrpc,
        on_notification=on_notification,
        on_request=on_request,
        on_unknown_frame=on_unknown,
    )
    async with peer.transport, asyncio.timeout(8):
        assert await peer.request("turn", {"text": "go"}) == "turn reply"
        await approval_started.wait()
        await notice_started.wait()
        assert notices == [] and unknown == []
        assert answers == [("approve", {"name": "tool"}, "0"), ("unknown", None, 0)]
        release_notice.set()
        release_approval.set()
        await checked.wait()
        assert notices == ["first", "second", "checked"]
        assert [(type(frame["id"]), frame["id"]) for frame in unknown] == [
            (str, "0"),
            (bool, False),
        ]
        await peer.notify("done")
        with pytest.raises(JsonRpcError) as raised:
            await peer.request("fail")
        assert (raised.value.code, raised.value.message, raised.value.data) == (
            -32000,
            "Call failed",
            {"detail": 7},
        )
    assert peer.transport.returncode == 0


async def test_rpc_unhandled_startup_request_gets_method_not_found() -> None:
    checked = asyncio.Event()

    async def notification(method: str, params: object) -> None:
        assert method == "checked"
        checked.set()

    peer = await JsonRpcPeer.start(
        child("""
            import json
            request = {'jsonrpc': '2.0', 'id': 'startup', 'method': 'unsupported'}
            print(json.dumps(request), flush=True)
            reply = json.loads(input())
            assert reply['id'] == 'startup' and reply['error']['code'] == -32601
            print(json.dumps({'jsonrpc': '2.0', 'method': 'checked'}), flush=True)
        """),
        on_notification=notification,
    )
    async with peer.transport, asyncio.timeout(8):
        await checked.wait()
    assert peer.transport.returncode == 0


@pytest.mark.parametrize("exit_early", [True, False])
async def test_rpc_pending_request_fails_on_eof_or_close(exit_early: bool) -> None:
    ready = asyncio.Event()

    async def notification(method: str, params: object) -> None:
        assert method == "ready"
        ready.set()

    peer = await JsonRpcPeer.start(
        child(
            """
            import sys
            input()
            print('{"method":"ready"}', flush=True)
            if sys.argv[1] == 'stay':
                sys.stdin.read()
        """,
            "exit" if exit_early else "stay",
        ),
        on_notification=notification,
    )
    async with peer.transport, asyncio.timeout(8):
        request = asyncio.create_task(peer.request("pending"))
        await ready.wait()
        if not exit_early:
            await peer.close()
        with pytest.raises(TransportError, match="closed"):
            await request


async def test_rpc_notification_backlog_fails_pending_requests() -> None:
    ready = asyncio.Event()

    async def notification(method: str, params: object) -> None:
        assert method == "ready"
        ready.set()
        await asyncio.Future()

    peer = await JsonRpcPeer.start(
        child("""
            import json, sys
            assert json.loads(input())['method'] == 'pending'
            print('{"method":"ready"}')
            for line in sys.stdin:
                call = json.loads(line)
                assert call['method'] == 'emit'
                print('{"method":"update"}')
                print(json.dumps({'id': call['id'], 'result': 'sent'}))
        """),
        on_notification=notification,
        max_queue_frames=3,
    )
    async with peer.transport, asyncio.timeout(3):
        pending = asyncio.create_task(peer.request("pending"))
        try:
            await ready.wait()
            assert await peer.request("emit") == "sent"
            assert await peer.request("emit") == "sent"
            with pytest.raises(TransportError, match="Transport frame queue overflow"):
                await peer.request("emit")
            with pytest.raises(TransportError, match="Transport frame queue overflow"):
                await pending
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_rpc_timeout_does_not_reuse_request_id() -> None:
    peer = await JsonRpcPeer.start(
        child("""
            import json
            first = json.loads(input())
            second = json.loads(input())
            assert first['id'] != second['id']
            print(json.dumps({'id': first['id'], 'result': 'old'}), flush=True)
            print(json.dumps({'id': second['id'], 'result': 'new'}), flush=True)
        """),
    )
    async with peer.transport, asyncio.timeout(8):
        with pytest.raises(TimeoutError):
            await peer.request("slow", timeout_s=0.1)
        assert await peer.request("fast") == "new"


async def test_rpc_close_delivers_final_notifications() -> None:
    notices: list[str] = []

    async def notification(method: str, params: object) -> None:
        await asyncio.sleep(0)
        notices.append(method)

    peer = await JsonRpcPeer.start(
        child("""
            import sys
            sys.stdin.read()
            print('{"method":"first"}', flush=True)
            print('{"method":"second"}', flush=True)
        """),
        on_notification=notification,
    )
    assert await peer.close(grace_s=0.5, term_s=0.2) == 0
    assert notices == ["first", "second"]


async def test_rpc_notification_can_close_its_own_transport() -> None:
    exit_codes: asyncio.Queue[int | None] = asyncio.Queue()

    async def notification(method: str, params: object) -> None:
        await exit_codes.put(await peer.close(grace_s=0.5, term_s=0.2))

    peer = await JsonRpcPeer.start(
        child("""
            import sys
            print('{"method":"done"}', flush=True)
            sys.stdin.read()
        """),
        on_notification=notification,
    )
    async with peer.transport, asyncio.timeout(8):
        assert await exit_codes.get() == 0


async def test_close_bounds_pipe_drain_with_detached_helper(tmp_path: Path) -> None:
    release = tmp_path / "release"
    exited = tmp_path / "exited"
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    transport = await StdioTransport.start(
        child(
            """
            import subprocess, sys
            from textwrap import dedent

            helper = '''
                import os, sys, time
                from pathlib import Path
                release, exited = map(Path, sys.argv[1:])
                deadline = time.monotonic() + 5
                while not release.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                closed = []
                for fd in (1, 2):
                    try:
                        os.write(fd, b"{}")
                    except BrokenPipeError:
                        closed.append(str(fd))
                result = exited.with_suffix(".tmp")
                result.write_text(",".join(closed))
                result.replace(exited)
            '''
            subprocess.Popen([sys.executable, '-c', dedent(helper), *sys.argv[1:]],
                             start_new_session=True)
            print('{"ready":true}', flush=True)
        """,
            str(release),
            str(exited),
        ),
        on_frame=frames.put,
    )
    try:
        async with asyncio.timeout(3):
            assert await frames.get() == {"ready": True}
        async with asyncio.timeout(1.5):
            assert await transport.close(grace_s=0.2, term_s=0.2) == 0
            assert not exited.exists()
    finally:
        release.touch()
        async with asyncio.timeout(8):
            await transport.close(grace_s=0.2, term_s=0.2)
            while not exited.exists():
                await asyncio.sleep(0.01)
    assert exited.read_text() == "1,2"


@pytest.mark.parametrize("parent_ignores_term", [False, True])
async def test_close_reaps_group_even_when_only_grandchild_is_left(
    parent_ignores_term: bool,
) -> None:
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    transport = await StdioTransport.start(
        child(
            """
            import signal
            import subprocess
            import sys
            from textwrap import dedent
            if sys.argv[1] == 'stubborn':
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            grandchild = '''
                import json, os, signal
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                print(json.dumps({'parent': os.getppid(), 'child': os.getpid()}), flush=True)
                for fd in (0, 1, 2):
                    os.close(fd)
                while True:
                    signal.pause()
            '''
            subprocess.Popen([sys.executable, '-c', dedent(grandchild)])
            sys.stdin.read()
            if sys.argv[1] == 'stubborn':
                while True:
                    signal.pause()
        """,
            "stubborn" if parent_ignores_term else "graceful",
        ),
        on_frame=frames.put,
    )
    async with transport, asyncio.timeout(8):
        pids = await frames.get()
        assert pids["parent"] == transport.pid
        first = asyncio.create_task(transport.close(grace_s=0.2, term_s=0.2))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        expected = -signal.SIGKILL if parent_ignores_term else 0
        assert await transport.close(grace_s=0.2, term_s=0.2) == expected
        assert await transport.close() == expected
        for pid in pids.values():
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        with pytest.raises(ProcessLookupError):
            os.killpg(transport.pid, 0)


@pytest.mark.parametrize("after_exit", [True, False])
async def test_close_handles_eperm_from_killpg_on_non_linux_posix(
    after_exit: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    source = (
        "print('{\"ready\":true}', flush=True)"
        if after_exit
        else "import signal\nprint('{\"ready\":true}', flush=True)\nsignal.pause()"
    )
    transport = await StdioTransport.start(child(source), on_frame=frames.put)
    real_killpg = os.killpg
    killpg_calls: list[int] = []

    def eperm_killpg(pgid: int, sig: int) -> None:
        if pgid == transport.pid:
            killpg_calls.append(sig)
            raise PermissionError(1, "Operation not permitted")
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", eperm_killpg)
    caplog.set_level("ERROR", logger="john.transport")
    async with asyncio.timeout(5):
        assert await frames.get() == {"ready": True}
        code = await transport.close(grace_s=0.05, term_s=0.1)
    assert code == (0 if after_exit else -signal.SIGTERM)
    assert killpg_calls
    assert "Process cleanup failed" not in caplog.text


async def test_close_bounds_non_linux_group_drain_after_sigkill(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("john.transport._KILL_DRAIN_S", 0.05)
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    transport = await StdioTransport.start(
        child("print('{\"ready\":true}', flush=True)"),
        on_frame=frames.put,
    )
    real_killpg = os.killpg
    signals_seen: list[int] = []

    def lingering_group(pgid: int, sig: int) -> None:
        if pgid == transport.pid:
            signals_seen.append(sig)
            return
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", lingering_group)
    caplog.set_level("WARNING", logger="john.transport")
    async with asyncio.timeout(2):
        assert await frames.get() == {"ready": True}
        assert await transport.close(grace_s=0.02, term_s=0.02) == 0
    assert signal.SIGTERM in signals_seen and signal.SIGKILL in signals_seen
    assert "still active after SIGKILL" in caplog.text
