"""paramiko-based SSH/SFTP client.

The pod's SSH endpoint comes from PodStatus.ssh_connection (string of form
`ssh user@host -p PORT` per Prime docs) or an object with explicit fields. We
parse defensively to handle both.
"""
from __future__ import annotations

import os
import posixpath
import re
import shlex
import stat
import time
from fnmatch import fnmatch

# Brief pause after API reports ACTIVE + ssh_connection — reduces immediate
# auth_propagation churn while sshd / keys settle on some providers.
SSH_POST_READY_SLEEP_S = 3.0
# One-time hint when auth_propagation persists — likely Prime/provider key injection.
SSH_AUTH_PROPAGATION_HINT_AFTER_S = 60.0
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

import paramiko


class SshRetryDetailCallback(Protocol):
    def __call__(self, attempt: int, total: int, delay_s: float, detail: str) -> None: ...


class SshAuthPropagationTimeout(RuntimeError):
    """SSH key auth kept failing long enough to try another provider."""


@dataclass
class SshEndpoint:
    host: str
    port: int
    user: str
    key_path: Path | None = None  # None → paramiko tries default keys + agent

    @classmethod
    def parse(cls, raw: str | dict | object, key_path: Path | None = None) -> "SshEndpoint":
        if raw is None:
            raise ValueError("ssh_connection is None — pod may still be provisioning")

        # Some providers return a one-element list, e.g. ["ubuntu@1.2.3.4"].
        if isinstance(raw, (list, tuple)):
            if not raw:
                raise ValueError("ssh_connection list is empty — pod may still be provisioning")
            return cls.parse(raw[0], key_path=key_path)

        # Dict shape
        if isinstance(raw, dict):
            host = raw.get("host") or raw.get("ip")
            port = int(raw.get("port") or 22)
            user = raw.get("user") or raw.get("username") or "root"
            if not host:
                raise ValueError(f"ssh_connection dict missing host: {raw!r}")
            return cls(host=host, port=port, user=user, key_path=key_path)

        # String shape: e.g. "ssh root@1.2.3.4 -p 12345" or "root@1.2.3.4:12345"
        if isinstance(raw, str):
            s = raw.strip()
            # Form: ssh user@host -p port
            m = re.match(
                r"^(?:ssh\s+)?(?P<user>[^@\s]+)@(?P<host>[^\s:]+)(?:\s+-p\s+(?P<port>\d+)|:(?P<port2>\d+))?",
                s,
            )
            if m:
                return cls(
                    host=m.group("host"),
                    port=int(m.group("port") or m.group("port2") or 22),
                    user=m.group("user"),
                    key_path=key_path,
                )
            raise ValueError(f"Could not parse ssh_connection string: {s!r}")

        # Pydantic model with attributes
        host = getattr(raw, "host", None) or getattr(raw, "ip", None)
        port = int(getattr(raw, "port", None) or 22)
        user = getattr(raw, "user", None) or getattr(raw, "username", None) or "root"
        if host:
            return cls(host=host, port=port, user=user, key_path=key_path)
        raise ValueError(f"Unrecognized ssh_connection shape: {type(raw).__name__}")


def parse_ssh_endpoint(raw: str | dict | object) -> SshEndpoint:
    """Parse Prime's ssh_connection value and attach the configured private key."""
    from primejob.auth import resolve_ssh_key_path

    return SshEndpoint.parse(raw, key_path=resolve_ssh_key_path())


def wait_for_ssh_connect(
    endpoint: SshEndpoint,
    *,
    max_wait_s: float = 300.0,
    retry_delay_s: float = 5.0,
    connect_timeout: float = 10.0,
    auth_warmup_s: float = 300.0,
    pod_ready_monotonic: float | None = None,
    on_retry: SshRetryDetailCallback | None = None,
    auth_failure_hint: str | None = None,
    auth_timeout_s: float | None = None,
) -> paramiko.SSHClient:
    """Block until SSH accepts our key or timeouts expire.

    Separates \"sshd still starting\" (transport errors) from auth failures during
    the post-boot propagation window on some providers.
    """
    deadline = time.monotonic() + max_wait_s
    total_approx = max(1, int(max_wait_s / retry_delay_s))
    pod_start = pod_ready_monotonic if pod_ready_monotonic is not None else time.monotonic()
    attempt = 0
    last_exc: Exception | None = None

    while time.monotonic() < deadline:
        attempt += 1
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = dict(
            hostname=endpoint.host,
            port=endpoint.port,
            username=endpoint.user,
            timeout=connect_timeout,
            allow_agent=True,
            look_for_keys=True,
        )
        if endpoint.key_path:
            kwargs["key_filename"] = str(endpoint.key_path)
        detail = "transport"
        try:
            client.connect(**kwargs)
            return client
        except paramiko.AuthenticationException as e:
            last_exc = e
            elapsed_pod = time.monotonic() - pod_start
            if auth_timeout_s is not None and elapsed_pod >= auth_timeout_s:
                raise SshAuthPropagationTimeout(
                    "SSH authentication kept failing during key propagation "
                    f"for ~{elapsed_pod:.0f}s."
                ) from e
            if elapsed_pod < auth_warmup_s:
                detail = "auth_propagation"
            else:
                hint = auth_failure_hint or (
                    "Run `primejob doctor` to verify your SSH key is registered in Prime."
                )
                raise RuntimeError(
                    "SSH authentication failed after waiting for key propagation. "
                    f"{hint}"
                ) from e
        except (paramiko.SSHException, OSError, TimeoutError) as e:
            last_exc = e
            detail = "transport"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_for = min(retry_delay_s, remaining)
        if on_retry is not None:
            on_retry(attempt, total_approx, sleep_for, detail)
        time.sleep(sleep_for)

    raise RuntimeError(
        f"SSH connect gave up after ~{max_wait_s:.0f}s to "
        f"{endpoint.user}@{endpoint.host}:{endpoint.port}: {last_exc}"
    )


class RetryCallback(Protocol):
    def __call__(self, attempt: int, total: int, delay_s: float) -> None: ...


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    def check(self) -> "ExecResult":
        if self.exit_code != 0:
            raise RuntimeError(
                f"Remote command failed (exit={self.exit_code}): {self.stderr or self.stdout}"
            )
        return self


class SshClient:
    """Context-managed paramiko SSH client with SFTP helpers."""

    def __init__(
        self,
        endpoint: SshEndpoint,
        *,
        connect_timeout: float = 10.0,
        retries: int = 24,
        retry_delay: float = 5.0,
        on_retry: RetryCallback | None = None,
        prec_connected: paramiko.SSHClient | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.connect_timeout = connect_timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.on_retry = on_retry
        self._prec_connected = prec_connected
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    def __enter__(self) -> "SshClient":
        self.connect(on_retry=self.on_retry)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self, *, on_retry: RetryCallback | None = None) -> None:
        if self._prec_connected is not None:
            self._client = self._prec_connected
            self._prec_connected = None
            return

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                kwargs = dict(
                    hostname=self.endpoint.host,
                    port=self.endpoint.port,
                    username=self.endpoint.user,
                    timeout=self.connect_timeout,
                    allow_agent=True,
                    look_for_keys=True,
                )
                if self.endpoint.key_path:
                    kwargs["key_filename"] = str(self.endpoint.key_path)
                client.connect(**kwargs)
                self._client = client
                return
            except (paramiko.SSHException, OSError) as e:
                last_exc = e
                if attempt < self.retries - 1:
                    if on_retry is not None:
                        on_retry(attempt + 1, self.retries, self.retry_delay)
                    time.sleep(self.retry_delay)
        raise RuntimeError(
            f"SSH connect failed after {self.retries} attempts to "
            f"{self.endpoint.user}@{self.endpoint.host}:{self.endpoint.port}: {last_exc}"
        )

    def close(self) -> None:
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:  # noqa: BLE001
                pass
            self._sftp = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def _require(self) -> paramiko.SSHClient:
        if self._client is None:
            raise RuntimeError("SshClient not connected — use as a context manager or call .connect()")
        return self._client

    @property
    def sftp(self) -> paramiko.SFTPClient:
        if self._sftp is None:
            self._sftp = self._require().open_sftp()
        return self._sftp

    def exec(self, cmd: str, *, env: dict[str, str] | None = None) -> ExecResult:
        """Run a command, return (exit_code, stdout, stderr) when done."""
        full_cmd = _with_env_prefix(cmd, env)
        stdin, stdout, stderr = self._require().exec_command(full_cmd, get_pty=False)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        return ExecResult(exit_code=code, stdout=out, stderr=err)

    def exec_stream(
        self,
        cmd: str,
        *,
        env: dict[str, str] | None = None,
        on_line: Callable[[str, str], None] | None = None,
    ) -> int:
        """Run a command and stream stdout/stderr line-by-line via callback.

        on_line(stream_name, line) — stream_name in {'stdout', 'stderr'}.
        Returns the remote exit code.
        """
        full_cmd = _with_env_prefix(cmd, env)
        transport = self._require().get_transport()
        if transport is None:
            raise RuntimeError("No active SSH transport")
        chan = transport.open_session()
        chan.exec_command(full_cmd)
        chan.set_combine_stderr(False)

        stdout_buf = b""
        stderr_buf = b""
        while True:
            done = chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready()
            if chan.recv_ready():
                stdout_buf += chan.recv(65536)
                stdout_buf = _emit_lines(stdout_buf, "stdout", on_line)
            if chan.recv_stderr_ready():
                stderr_buf += chan.recv_stderr(65536)
                stderr_buf = _emit_lines(stderr_buf, "stderr", on_line)
            if done:
                break
            time.sleep(0.05)
        # flush trailing partials
        if stdout_buf and on_line:
            on_line("stdout", stdout_buf.decode("utf-8", errors="replace"))
        if stderr_buf and on_line:
            on_line("stderr", stderr_buf.decode("utf-8", errors="replace"))
        return chan.recv_exit_status()

    def mkdir_p(self, remote_path: str) -> None:
        parts = remote_path.strip("/").split("/")
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}" if cur else f"/{p}"
            try:
                self.sftp.stat(cur)
            except FileNotFoundError:
                self.sftp.mkdir(cur)

    def upload(
        self,
        local_path: str | Path,
        remote_path: str,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        local = Path(local_path)
        if local.is_dir():
            self._upload_dir(local, remote_path, progress=progress)
        else:
            parent = posixpath.dirname(remote_path)
            if parent:
                self.mkdir_p(parent)
            self.sftp.put(str(local), remote_path, callback=progress)

    def _upload_dir(self, local: Path, remote: str, progress=None) -> None:
        self.mkdir_p(remote)
        for entry in sorted(local.rglob("*")):
            rel = entry.relative_to(local).as_posix()
            target = posixpath.join(remote, rel)
            if entry.is_dir():
                self.mkdir_p(target)
            else:
                self.sftp.put(str(entry), target, callback=progress)

    def download(
        self,
        remote_path: str,
        local_path: str | Path,
        *,
        ignore_permission_denied: bool = False,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        match_root_name: str | None = None,
    ) -> None:
        """Recursive download. If remote is a directory, mirror it into local."""
        self._download(
            remote_path,
            Path(local_path),
            ignore_permission_denied=ignore_permission_denied,
            include_patterns=include_patterns or [],
            exclude_patterns=exclude_patterns or [],
            match_root_name=match_root_name,
            rel_path="",
        )

    def _download(
        self,
        remote_path: str,
        local: Path,
        *,
        ignore_permission_denied: bool,
        include_patterns: list[str],
        exclude_patterns: list[str],
        match_root_name: str | None,
        rel_path: str,
    ) -> bool:
        try:
            st = self.sftp.stat(remote_path)
        except PermissionError:
            if ignore_permission_denied:
                return False
            raise
        except FileNotFoundError:
            return False
        if stat.S_ISDIR(st.st_mode):
            try:
                entries = self.sftp.listdir_attr(remote_path)
            except PermissionError:
                if ignore_permission_denied:
                    return False
                raise
            downloaded = False
            for entry in entries:
                child_rel = (
                    f"{rel_path}/{entry.filename}" if rel_path else entry.filename
                )
                child_downloaded = self._download(
                    posixpath.join(remote_path, entry.filename),
                    local / entry.filename,
                    ignore_permission_denied=ignore_permission_denied,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    match_root_name=match_root_name,
                    rel_path=child_rel,
                )
                downloaded = downloaded or child_downloaded
            if downloaded or not include_patterns and not exclude_patterns:
                local.mkdir(parents=True, exist_ok=True)
            return downloaded
        else:
            match_path = _download_match_path(rel_path, match_root_name)
            if not _download_path_selected(match_path, include_patterns, exclude_patterns):
                return False
            local.parent.mkdir(parents=True, exist_ok=True)
            self.sftp.get(remote_path, str(local))
            return True


def exec_oneshot(endpoint: SshEndpoint, cmd: str, *, timeout: float = 10.0) -> ExecResult:
    """Open a fresh SSH connection, run one command, close. Suitable for periodic
    polling (e.g. nvidia-smi) where we don't want to share the main streaming channel."""
    with SshClient(endpoint, connect_timeout=timeout, retries=2, retry_delay=1.0) as sh:
        return sh.exec(cmd)


def _emit_lines(buf: bytes, stream: str, cb: Callable[[str, str], None] | None) -> bytes:
    if cb is None:
        return b""
    while b"\n" in buf:
        line, _, buf = buf.partition(b"\n")
        cb(stream, line.decode("utf-8", errors="replace"))
    return buf


def _with_env_prefix(cmd: str, env: dict[str, str] | None) -> str:
    if not env:
        return cmd
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"env {exports} sh -c {shlex.quote(cmd)}"


def _download_match_path(rel_path: str, root_name: str | None) -> str:
    return f"{root_name}/{rel_path}" if root_name and rel_path else rel_path


def _download_path_selected(
    rel_path: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> bool:
    included = not include_patterns or any(fnmatch(rel_path, p) for p in include_patterns)
    excluded = any(fnmatch(rel_path, p) for p in exclude_patterns)
    return included and not excluded
