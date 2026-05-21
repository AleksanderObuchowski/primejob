"""End-to-end orchestration of `primejob run`.

The big idea:
  1. resolve config + pick cheapest GPU offering
  2. confirm cost (via sink — plain prompt or TUI modal)
  3. ensure disk (if configured)
  4. create pod, wait for SSH
  5. tarball cwd, upload, unpack on pod
  6. uv sync + uv run python <script> with env forwarded, stream output
  7. download outputs/ back, terminate pod no matter what
"""
from __future__ import annotations

import os
import re
import shutil
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values
from prime_cli.api.client import APIClient

from primejob.auth import (
    check_ssh_key,
    require_ssh_key,
    ssh_auth_failure_hint,
)
from primejob.backend.disks import disk_location, ensure_disk, find_disk, wait_for_disk_detached
from primejob.backend.pods import (
    PodSpec,
    create_pod,
    get_pod,
    mount_path_for_disk,
    terminate,
    wait_for_running,
)
from primejob.backend.ssh import SshClient, parse_ssh_endpoint, wait_for_ssh_connect
from primejob.config import ProjectConfig, load_project_config
from primejob.events import ConfirmRequest, ConsoleSink, EventSink
from primejob.packaging import make_tarball
from primejob.pricing import pick_cheapest, resolve_gpu_type, normalize_provider_name
from primejob.runtime import CleanupGuard, CostTracker, StatusBar
from primejob.state import RunRecord, new_run_id
from primejob.tui.state import FinalSummary, Phase, RunMeta


REMOTE_WORK = "/tmp/primejob/work"
REMOTE_TARBALL = "/tmp/primejob/src.tar.gz"
REMOTE_BIN = "/tmp/primejob/bin"
REMOTE_UV = f"{REMOTE_BIN}/uv"
REMOTE_DATASET = "/tmp/primejob/dataset"

# Brief pause after API reports ACTIVE + ssh_connection — reduces immediate
# auth_propagation churn while sshd / keys settle on some providers.
SSH_POST_READY_SLEEP_S = 3.0
# One-time hint when auth_propagation persists — likely Prime/provider key injection.
SSH_AUTH_PROPAGATION_HINT_AFTER_S = 60.0

DATA_MODES = frozenset({"attach", "stage", "none", "local"})


@dataclass
class RunOptions:
    script: str
    args: list[str] = field(default_factory=list)
    gpu: str | None = None
    count: int = 1
    country: str | None = None
    disk: str | None = None
    yes: bool = False
    disk_size_gb: int | None = None
    data_mode: str = "attach"  # attach | stage | none | local
    data_subdir: str | None = None
    include_data: list[str] = field(default_factory=list)
    setup_ssh: bool = False
    skip_providers: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    record: RunRecord


class RunAborted(Exception):
    pass


def run_training(
    client: APIClient,
    opts: RunOptions,
    *,
    project: ProjectConfig | None = None,
    cwd: Path | None = None,
    sink: EventSink | None = None,
) -> RunResult:
    cwd = cwd or Path.cwd()
    project = project or load_project_config(cwd)

    own_sink = False
    if sink is None:
        # Default plain-mode sink writes to ~/.primejob/runs/<id>/log.txt — but
        # we don't know run_id yet. Open a temp sink that doesn't tee to a file
        # and reopen later once we have a RunRecord.
        sink = ConsoleSink(yes=opts.yes)
        own_sink = True

    sink.phase(Phase.PREFLIGHT)
    _validate_workspace(cwd, project)
    if opts.setup_ssh:
        from rich.console import Console

        from primejob.onboarding import ensure_ssh_key_ready

        ensure_ssh_key_ready(
            client,
            Console(stderr=True),
            assume_yes=True,
            interactive=False,
        )
    require_ssh_key(client)

    gpu_type = resolve_gpu_type(opts.gpu or project.default_gpu)
    country = opts.country or project.default_country
    data_mode = opts.data_mode.lower()
    if data_mode not in DATA_MODES:
        raise RuntimeError(f"data_mode must be one of: {', '.join(sorted(DATA_MODES))}")

    if data_mode in {"none", "local"}:
        disk_name = None
    else:
        disk_name = opts.disk or project.dataset_disk

    bundle_paths = _resolve_bundle_paths(cwd, opts, project, data_mode)

    if data_mode == "stage" and not disk_name:
        raise RuntimeError("--data-mode stage requires --disk or [tool.primejob].dataset_disk")
    disk_id = None

    existing = find_disk(client, disk_name) if disk_name else None
    if existing is not None:
        disk_id = existing.id
        disk_country, _, _ = disk_location(existing)
        # Attach mode must place the training pod where this disk can mount.
        if data_mode == "attach" and not country:
            country = disk_country

    sink.status(f"Picking cheapest {gpu_type} ×{opts.count} (country={country or 'any'})...")
    exclude_providers = _merged_exclude_providers(project, opts.skip_providers)
    if exclude_providers:
        sink.status_note(
            f"  Skipping providers: {', '.join(sorted(exclude_providers))}"
        )
    option = pick_cheapest(
        client,
        gpu_type=gpu_type,
        gpu_count=opts.count,
        country=country,
        disks=[disk_id] if disk_id and data_mode == "attach" else None,
        exclude_providers=exclude_providers,
    )
    rate = option.effective_price
    sink.status(
        f"  → {option.gpu_type} ×{option.gpu_count} @ {option.provider} "
        f"({option.country}, {option.data_center}) ${rate:.4f}/h"
    )

    if not opts.yes:
        prompt = f"Spawn pod at ${rate:.4f}/h? [y/N]: "
        ok = sink.confirm(ConfirmRequest(
            prompt=prompt,
            gpu_type=option.gpu_type,
            gpu_count=option.gpu_count,
            rate_per_hr=rate,
            provider=option.provider,
            country=option.country,
        ))
        if not ok:
            raise RunAborted("User declined.")

    if disk_name and disk_id is None:
        sink.status(f"Ensuring disk '{disk_name}'...")
        # If we have to create it, use the offering's country.
        disk = ensure_disk(
            client,
            name=disk_name,
            size_gb=opts.disk_size_gb or project.default_disk_size,
            country=country or option.country,
            wait=True,
        )
        disk_id = disk.id

    run_id = new_run_id()
    staged_dataset_path: Path | None = None
    record = RunRecord(
        run_id=run_id,
        pod_id=None,
        gpu_type=option.gpu_type,
        gpu_count=option.gpu_count,
        country=option.country,
        provider=option.provider,
        rate_per_hr=rate,
        script=opts.script,
        args=list(opts.args),
        disk_name=disk_name,
    )
    record.ensure_dir()

    # Now that run_id exists, swap in a sink that also tees to the log file
    # (only for ConsoleSink we own — external sinks manage their own files).
    if own_sink:
        try:
            sink.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        sink = ConsoleSink(log_file=record.log_path, yes=opts.yes)
        own_sink = True

    sink.meta(RunMeta(
        run_id=run_id,
        script=opts.script,
        args=list(opts.args),
        gpu_type=option.gpu_type,
        gpu_count=option.gpu_count,
        country=option.country,
        provider=option.provider,
    ))

    if disk_name and data_mode == "stage":
        from primejob.dataset import pull

        staged_dataset_path = cwd / ".primejob" / "staged" / run_id / "dataset"
        if staged_dataset_path.exists():
            shutil.rmtree(staged_dataset_path)

        def stage_progress(status):
            state = (status.status or "?").lower()
            sink.status_note(f"  staging-pod={state} install={status.installation_progress or 0}%")

        sink.status(
            f"Staging dataset from disk '{disk_name}'"
            f"{f'/{opts.data_subdir}' if opts.data_subdir else ''}..."
        )
        pulled = pull(
            client,
            disk_name=disk_name,
            local_path=staged_dataset_path,
            country=country,
            subdir=opts.data_subdir,
            on_progress=stage_progress,
        )
        mb = pulled.bytes_downloaded / (1024 ** 2)
        sink.status(
            f"  → staged {pulled.files_downloaded} files, {mb:.1f} MB "
            f"in {pulled.elapsed_s:.1f}s"
        )

    sink.phase(Phase.PROVISION)
    sink.status(f"Creating pod (run_id={run_id})...")
    spec = PodSpec(
        name=f"primejob-{run_id}",
        gpu_option=option,
        disk_ids=[disk_id] if disk_id and data_mode == "attach" else None,
    )
    pod = create_pod(client, spec)
    record.pod_id = pod.id
    record.save()

    sink.meta(RunMeta(
        run_id=run_id,
        script=opts.script,
        args=list(opts.args),
        gpu_type=option.gpu_type,
        gpu_count=option.gpu_count,
        country=option.country,
        provider=option.provider,
        pod_id=pod.id,
    ))

    tracker = CostTracker(rate_per_hr=rate)
    error_lines: list[str] = []
    failed_phase: Phase | None = None

    def cleanup() -> None:
        if record.pod_id:
            sink.status(f"Terminating pod {record.pod_id}...")
            try:
                terminate(client, record.pod_id)
            except Exception as e:  # noqa: BLE001
                sink.status(f"  (terminate failed: {e})")
        if disk_id and data_mode == "attach":
            try:
                sink.status(f"Waiting for disk '{disk_name}' to detach...")
                wait_for_disk_detached(client, disk_id, timeout=180)
            except Exception as e:  # noqa: BLE001
                sink.status(f"  (disk detach wait failed: {e})")
        if record.ended_at is None:
            record.ended_at = time.time()
            record.total_cost = tracker.spent()
            if record.status == "running":
                record.status = "terminated"
            record.save()

    with CleanupGuard(cleanup):
        sink.status("Waiting for pod to become running...")

        def progress(status):
            state = (status.status or "?").lower()
            sink.status_note(
                f"  pod={state} install={status.installation_progress or 0}% "
                f"rate=${status.cost_per_hr or 0:.4f}/h"
            )
            if status.cost_per_hr:
                tracker.update_rate(status.cost_per_hr)
            sink.cost(
                started_at=tracker.started_at,
                rate_per_hr=tracker.rate_per_hr,
                spent=tracker.spent(),
            )

        try:
            pod_status = wait_for_running(client, pod.id, on_progress=progress)
            fresh = get_pod(client, pod.id)
            ssh = parse_ssh_endpoint(pod_status.ssh_connection)
            sink.status(
                f"Pod reachable ({fresh.status or '?'}); connecting SSH to "
                f"{ssh.user}@{ssh.host}:{ssh.port}…"
            )
            if SSH_POST_READY_SLEEP_S > 0:
                sink.status_note(
                    f"  Waiting {SSH_POST_READY_SLEEP_S:.0f}s before first SSH attempt…"
                )
                time.sleep(SSH_POST_READY_SLEEP_S)

            pod_ready_mono = time.monotonic()
            ssh_status = check_ssh_key(client)
            auth_hint = ssh_auth_failure_hint(ssh_status, provider=option.provider)
            auth_hint_shown = False

            def ssh_retry_detail(
                attempt: int, total: int, delay_s: float, detail: str
            ) -> None:
                nonlocal auth_hint_shown
                sink.status_note(
                    f"  SSH [{detail}] attempt {attempt}/{total}, retry in {delay_s:.0f}s…"
                )
                if (
                    detail == "auth_propagation"
                    and not auth_hint_shown
                    and time.monotonic() - pod_ready_mono >= SSH_AUTH_PROPAGATION_HINT_AFTER_S
                ):
                    auth_hint_shown = True
                    sink.status_note(
                        "  [hint] Prime may not have injected your registered primary SSH "
                        f"key into this provider ({option.provider or 'unknown'}). "
                        "Try `--skip-provider`, `[tool.primejob].exclude_providers`, "
                        "or a different `--country`."
                    )

            auth_window = min(float(project.ssh_max_wait), 300.0)
            connected = wait_for_ssh_connect(
                ssh,
                max_wait_s=float(project.ssh_max_wait),
                retry_delay_s=float(project.ssh_retry_delay),
                pod_ready_monotonic=pod_ready_mono,
                auth_warmup_s=auth_window,
                on_retry=ssh_retry_detail,
                auth_failure_hint=auth_hint,
            )
            sink.status(f"SSH connected as {ssh.user}@{ssh.host}:{ssh.port}")
            sink.ssh_ready(ssh)
        except Exception:
            failed_phase = Phase.PROVISION
            sink.phase(Phase.PROVISION, failed=True)
            raise

        tarball = cwd / ".primejob" / "src.tar.gz"

        bar = StatusBar(
            run_id,
            tracker,
            lambda msg: sink.status_note(msg),
            interval=30.0,
            on_tick=lambda: sink.cost(
                started_at=tracker.started_at,
                rate_per_hr=tracker.rate_per_hr,
                spent=tracker.spent(),
            ),
        )

        try:
            with SshClient(ssh, prec_connected=connected) as sh:
                sink.phase(Phase.UPLOAD)
                if bundle_paths:
                    sink.status(
                        "Packaging local src (respecting .gitignore, bundling extra data paths)..."
                    )
                else:
                    sink.status("Packaging local src (respecting .gitignore)...")
                tar = make_tarball(cwd, tarball, extra_paths=bundle_paths or None)
                sink.status(f"  → {tar.file_count} files, {tar.bytes_size/1024/1024:.1f} MB")

                env = _build_remote_env(cwd, project.forward_env)

                sink.status("Uploading src tarball...")
                try:
                    sh.upload(tarball, REMOTE_TARBALL)
                    sh.exec(
                        f"mkdir -p {REMOTE_WORK} && tar -xzf {REMOTE_TARBALL} -C {REMOTE_WORK} && rm -f {REMOTE_TARBALL}"
                    ).check()
                except Exception:
                    failed_phase = Phase.UPLOAD
                    sink.phase(Phase.UPLOAD, failed=True)
                    raise

                disk_mount = (
                    mount_path_for_disk(fresh, disk_id) if disk_id and data_mode == "attach" else None
                )
                if disk_mount:
                    sink.status(f"Persistent disk mounted at {disk_mount}")
                    env["PRIMEJOB_DATASET_PATH"] = disk_mount
                elif staged_dataset_path is not None:
                    sink.status(f"Uploading staged dataset to {REMOTE_DATASET}...")
                    sh.upload(staged_dataset_path, REMOTE_DATASET)
                    env["PRIMEJOB_DATASET_PATH"] = REMOTE_DATASET
                    try:
                        shutil.rmtree(staged_dataset_path.parent)
                    except Exception:  # noqa: BLE001
                        pass
                elif data_mode == "local" and bundle_paths:
                    local_dataset = _remote_dataset_path_for_bundle(cwd, bundle_paths[0])
                    sink.status(f"Bundled dataset available at {local_dataset}")
                    env["PRIMEJOB_DATASET_PATH"] = local_dataset

                sink.phase(Phase.INSTALL)
                sink.status("Installing uv on the pod...")
                install = sh.exec(
                    f"mkdir -p {shlex.quote(REMOTE_BIN)} && "
                    f"test -x {shlex.quote(REMOTE_UV)} || "
                    f"curl -LsSf https://astral.sh/uv/install.sh | "
                    f"env UV_INSTALL_DIR={shlex.quote(REMOTE_BIN)} sh"
                )
                if install.exit_code != 0:
                    for line in install.stdout.splitlines():
                        sink.log_line("stderr", line)
                    for line in install.stderr.splitlines():
                        sink.log_line("stderr", line)
                        error_lines.append(line)
                    failed_phase = Phase.INSTALL
                    sink.phase(Phase.INSTALL, failed=True)
                    raise RuntimeError(f"uv install failed (exit={install.exit_code})")

                sink.status("Running `uv sync`...")
                sync = sh.exec(
                    f"cd {shlex.quote(REMOTE_WORK)} && "
                    f"PATH={shlex.quote(REMOTE_BIN)}:$PATH {shlex.quote(REMOTE_UV)} sync"
                )
                if sync.exit_code != 0:
                    for line in sync.stdout.splitlines():
                        sink.log_line("stderr", line)
                    for line in sync.stderr.splitlines():
                        sink.log_line("stderr", line)
                        error_lines.append(line)
                    failed_phase = Phase.INSTALL
                    sink.phase(Phase.INSTALL, failed=True)
                    raise RuntimeError(f"uv sync failed (exit={sync.exit_code})")

                sink.phase(Phase.RUNNING)
                bar.start()
                cmd_args = " ".join(shlex.quote(a) for a in opts.args)
                remote_cmd = (
                    f"cd {shlex.quote(REMOTE_WORK)} && "
                    f"PATH={shlex.quote(REMOTE_BIN)}:$PATH "
                    f"{shlex.quote(REMOTE_UV)} run python {shlex.quote(opts.script)} {cmd_args}"
                )
                sink.status(f"Running: {opts.script} {cmd_args}".rstrip())

                def on_line(stream: str, line: str) -> None:
                    sink.log_line(stream, line)
                    if _is_error_line(line):
                        error_lines.append(line)

                exit_code = sh.exec_stream(remote_cmd, env=env, on_line=on_line)
                record.exit_code = exit_code

                sink.phase(Phase.WRAP)
                sink.status(f"Remote process exited with code {exit_code}")
                sink.status("Downloading outputs/...")
                local_outputs = cwd / "outputs" / run_id
                try:
                    sh.download(f"{REMOTE_WORK}/outputs", local_outputs)
                except FileNotFoundError:
                    sink.status("  (no outputs/ produced)")
        finally:
            bar.stop()

        record.ended_at = time.time()
        record.total_cost = tracker.spent()
        record.status = "finished" if record.exit_code == 0 else "failed"
        record.save()

        if record.status == "failed":
            sink.phase(Phase.RUNNING, failed=True)
        else:
            sink.phase(Phase.DONE)

        sink.status(
            f"Done. exit={record.exit_code} elapsed={tracker.elapsed()} "
            f"cost=${record.total_cost:.4f} outputs={cwd / 'outputs' / run_id}"
        )

    sink.finish(FinalSummary(
        exit_code=record.exit_code,
        status=record.status,
        elapsed_s=time.time() - record.started_at,
        total_cost=record.total_cost or 0.0,
        outputs_path=str(cwd / "outputs" / run_id),
        last_error=error_lines[-20:],
    ))

    if own_sink:
        try:
            sink.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    return RunResult(record=record)


def _merged_exclude_providers(
    project: ProjectConfig, cli_skip: list[str]
) -> list[str]:
    """Merge pyproject exclude_providers with CLI --skip-provider flags."""
    seen: set[str] = set()
    out: list[str] = []
    for name in [*project.exclude_providers, *cli_skip]:
        normalized = normalize_provider_name(name)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(name.strip())
    return out


def _resolve_bundle_paths(
    cwd: Path,
    opts: RunOptions,
    project: ProjectConfig,
    data_mode: str,
) -> list[Path]:
    if data_mode != "local":
        return []
    raw_paths = opts.include_data or project.bundle_paths
    if not raw_paths:
        raise RuntimeError(
            "--data-mode local requires --include-data PATH and/or "
            "[tool.primejob].bundle_paths in pyproject.toml"
        )
    resolved: list[Path] = []
    for raw in raw_paths:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (cwd / path).resolve()
        else:
            path = path.resolve()
        if cwd.resolve() not in path.parents and path != cwd.resolve():
            raise RuntimeError(f"Bundle path must live inside the project directory: {raw}")
        if not path.exists():
            raise RuntimeError(f"Bundle path not found: {raw}")
        resolved.append(path)
    return resolved


def _remote_dataset_path_for_bundle(cwd: Path, bundle_path: Path) -> str:
    rel = bundle_path.resolve().relative_to(cwd.resolve()).as_posix()
    return f"{REMOTE_WORK}/{rel}"


def _validate_workspace(cwd: Path, project: ProjectConfig) -> None:
    if not (cwd / "pyproject.toml").exists():
        raise RuntimeError(
            f"No pyproject.toml in {cwd}. primejob expects a uv-managed project."
        )
    if not (cwd / "uv.lock").exists():
        raise RuntimeError(
            f"No uv.lock in {cwd}. Run `uv lock` (or `uv sync`) before `primejob run`."
        )
    missing = []
    env_path = cwd / ".env"
    env_values = dotenv_values(env_path) if env_path.exists() else {}
    for key in project.forward_env:
        if key not in env_values and key not in os.environ:
            missing.append(key)
    if missing:
        raise RuntimeError(
            f"Missing env keys for forward_env: {missing}. "
            f"Put them in {env_path} or your shell environment."
        )


def _build_remote_env(cwd: Path, forward: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    env_path = cwd / ".env"
    env_values = dotenv_values(env_path) if env_path.exists() else {}
    for key in forward:
        if key in env_values and env_values[key] is not None:
            out[key] = env_values[key]
        elif key in os.environ:
            out[key] = os.environ[key]
    return out


_ERR_PATTERNS = re.compile(
    r"\b(Error|Exception|Traceback|FAILED|OOM|CUDA out of memory)\b",
    re.IGNORECASE,
)


def _is_error_line(line: str) -> bool:
    return bool(_ERR_PATTERNS.search(line))
