"""primejob CLI entry point."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from primejob import __version__
from primejob.auth import check_auth, get_client
from primejob.config import load_project_config
from primejob.pricing import list_gpus, resolve_gpu_type

console = Console()

app = typer.Typer(
    name="primejob",
    help="Run GPU training jobs on Prime Intellect.",
    no_args_is_help=True,
    add_completion=False,
)

gpus_app = typer.Typer(help="GPU marketplace inspection.", no_args_is_help=True)
app.add_typer(gpus_app, name="gpus")

dataset_app = typer.Typer(help="Persistent dataset disk operations.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")

runs_app = typer.Typer(help="Run history.", no_args_is_help=True)
app.add_typer(runs_app, name="runs")


def _not_implemented(name: str) -> None:
    typer.secho(f"[primejob] '{name}' not implemented yet", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


@app.command()
def login() -> None:
    """Wrap `prime login` for first-time auth."""
    import subprocess

    bin_path = shutil.which("prime")
    if not bin_path:
        console.print("[red]`prime` CLI not found on PATH.[/red] Install with `uv add prime` or pipx.")
        raise typer.Exit(code=1)
    raise typer.Exit(code=subprocess.run([bin_path, "login"]).returncode)


@app.command()
def doctor() -> None:
    """Verify auth, SDK, paramiko, prime CLI."""
    table = Table(title="primejob doctor", show_header=False, box=None)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")

    ok = "[green]ok[/green]"
    bad = "[red]fail[/red]"

    table.add_row("primejob version", ok, __version__)
    table.add_row("python", ok, sys.version.split()[0])

    try:
        import paramiko
        table.add_row("paramiko", ok, paramiko.__version__)
        paramiko_ok = True
    except Exception as e:  # noqa: BLE001
        table.add_row("paramiko", bad, str(e))
        paramiko_ok = False

    try:
        import prime_cli
        table.add_row("prime_cli", ok, getattr(prime_cli, "__version__", "installed"))
    except Exception as e:  # noqa: BLE001
        table.add_row("prime_cli", bad, str(e))

    prime_bin = shutil.which("prime")
    if prime_bin:
        table.add_row("prime CLI", ok, prime_bin)
    else:
        table.add_row("prime CLI", "[yellow]missing[/yellow]", "not on PATH (needed for `primejob login`)")

    auth = check_auth()
    auth_detail = []
    if auth.has_env_key:
        auth_detail.append("PRIME_API_KEY env")
    if auth.has_config_file:
        auth_detail.append(f"~/.prime/config.json")
    detail = ", ".join(auth_detail) or "no credentials"
    if auth.client_ok:
        table.add_row("auth", ok, detail)
    else:
        table.add_row("auth", bad, f"{detail} — {auth.error or 'unknown'}")

    console.print(table)

    if not (paramiko_ok and auth.client_ok):
        raise typer.Exit(code=1)


@gpus_app.command("list")
def gpus_list(
    country: str | None = typer.Option(None, "--country", "-c", help="ISO country code, e.g. US."),
    gpu: str | None = typer.Option(None, "--gpu", "-g", help="Filter by GPU type, e.g. H100."),
    count: int | None = typer.Option(None, "--count", "-n", help="Filter by GPU count."),
    limit: int = typer.Option(30, "--limit", help="Max rows to display."),
) -> None:
    """List available GPUs with prices."""
    client = get_client()
    resolved = resolve_gpu_type(gpu) if gpu else None
    if gpu and resolved != gpu:
        console.print(f"[dim]Filter '{gpu}' resolved to '{resolved}'[/dim]")
    options = list_gpus(client, country=country, gpu_type=gpu, gpu_count=count)

    if not options:
        console.print("[yellow]No offerings found for that filter.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="GPU availability (sorted by price)", show_lines=False)
    table.add_column("GPU", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Mem", justify="right")
    table.add_column("Country")
    table.add_column("Provider")
    table.add_column("On-demand", justify="right")
    table.add_column("Community", justify="right")
    table.add_column("Stock")

    for o in options[:limit]:
        od = f"${o.on_demand_price:.4f}/h" if o.on_demand_price else "-"
        cp = f"${o.community_price:.4f}/h" if o.community_price else "-"
        mem = f"{o.gpu_memory}GB" if o.gpu_memory else "-"
        table.add_row(
            o.gpu_type or "?",
            str(o.gpu_count or "?"),
            mem,
            o.country or "-",
            o.provider or "-",
            od,
            cp,
            o.stock_status or "-",
        )
    console.print(table)
    if len(options) > limit:
        console.print(f"[dim](showing top {limit} of {len(options)} — use --limit to show more)[/dim]")


def _resolve_disk_name(disk_arg: str | None) -> str:
    if disk_arg:
        return disk_arg
    cfg = load_project_config()
    if cfg.dataset_disk:
        return cfg.dataset_disk
    raise typer.BadParameter(
        "No disk name given and [tool.primejob] dataset_disk is unset in pyproject.toml. "
        "Pass --disk NAME or configure it."
    )


@dataset_app.command("push")
def dataset_push(
    local_path: str = typer.Argument(..., help="Local file or directory to upload."),
    disk: str | None = typer.Option(None, "--disk", "-d", help="Disk name (defaults to pyproject)."),
    size: int | None = typer.Option(None, "--size", help="Disk size in GB when creating fresh."),
    country: str | None = typer.Option(None, "--country", "-c", help="ISO country (only if creating disk)."),
    subdir: str | None = typer.Option(None, "--subdir", help="Remote subdirectory (default: basename of local path)."),
) -> None:
    """Upload a dataset to a persistent disk."""
    from primejob.dataset import push  # local import keeps CLI startup snappy

    disk_name = _resolve_disk_name(disk)
    src = Path(local_path).expanduser().resolve()

    def progress(status):
        state = (status.status or "?").lower()
        progress_pct = status.installation_progress or 0
        console.print(
            f"[dim]helper pod: state={state} install={progress_pct}% cost=${status.cost_per_hr or 0:.4f}/h[/dim]"
        )

    console.print(
        f"[bold]Pushing[/bold] {src} → disk [cyan]{disk_name}[/cyan] (creating if missing)..."
    )
    res = push(
        get_client(),
        disk_name=disk_name,
        local_path=src,
        disk_size_gb=size,
        country=country,
        subdir=subdir,
        on_progress=progress,
    )
    mb = res.bytes_uploaded / (1024 ** 2)
    console.print(
        f"[green]Done.[/green] {res.files_uploaded} files, {mb:.1f} MB in {res.elapsed_s:.1f}s "
        f"(pod {res.pod_id} terminated)"
    )


@dataset_app.command("list")
def dataset_list(
    disk: str | None = typer.Option(None, "--disk", "-d", help="Disk name (defaults to pyproject)."),
    country: str | None = typer.Option(None, "--country", "-c"),
) -> None:
    """List files on a persistent disk."""
    from primejob.dataset import list_files

    disk_name = _resolve_disk_name(disk)

    def progress(status):
        state = (status.status or "?").lower()
        console.print(f"[dim]helper pod: state={state}[/dim]")

    console.print(f"[bold]Listing[/bold] disk [cyan]{disk_name}[/cyan]...")
    files = list_files(get_client(), disk_name=disk_name, country=country, on_progress=progress)
    if not files:
        console.print("[yellow](empty disk)[/yellow]")
        return
    for f in files:
        console.print(f"  {f}")
    console.print(f"[dim]{len(files)} file(s)[/dim]")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    script: str = typer.Argument(..., help="Python script to run on the pod."),
    gpu: str | None = typer.Option(None, "--gpu", "-g", help="GPU type."),
    count: int = typer.Option(1, "--count", "-n", help="GPU count."),
    country: str | None = typer.Option(None, "--country", "-c", help="ISO country code."),
    disk: str | None = typer.Option(None, "--disk", "-d", help="Persistent disk to attach."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost confirmation."),
    disk_size: int | None = typer.Option(None, "--disk-size", help="Disk size in GB (if creating)."),
    data_mode: str = typer.Option(
        "attach",
        "--data-mode",
        help="Dataset handling: attach persistent disk to the training pod, or stage a local copy first.",
    ),
    data_subdir: str | None = typer.Option(
        None,
        "--data-subdir",
        help="Subdirectory on the dataset disk to stage (only with --data-mode stage).",
    ),
    plain: bool = typer.Option(False, "--plain", help="Force plain streaming output (no TUI)."),
    exit_on_finish: bool = typer.Option(False, "--exit-on-finish", help="In TUI mode, skip summary screen and exit immediately."),
) -> None:
    """Submit a training job to a Prime Intellect pod.

    Anything after the script is forwarded as args to the script.

    By default opens a Textual dashboard when stdout is a TTY. Pass --plain to
    force the old line-by-line streaming output (useful for CI / nohup / pipes).
    """
    from primejob.run import RunAborted, RunOptions, run_training

    opts = RunOptions(
        script=script,
        args=list(ctx.args),
        gpu=gpu,
        count=count,
        country=country,
        disk=disk,
        yes=yes,
        disk_size_gb=disk_size,
        data_mode=data_mode,
        data_subdir=data_subdir,
    )

    use_tui = sys.stdout.isatty() and not plain
    if use_tui:
        from primejob.tui import run_dashboard
        try:
            code = run_dashboard(get_client(), opts, exit_on_finish=exit_on_finish)
        except RunAborted as e:
            console.print(f"[yellow]Aborted:[/yellow] {e}")
            raise typer.Exit(code=130)
        raise typer.Exit(code=code)

    # Plain mode — default ConsoleSink in run_training preserves prior UX.
    try:
        result = run_training(get_client(), opts)
    except RunAborted as e:
        console.print(f"[yellow]Aborted:[/yellow] {e}")
        raise typer.Exit(code=130)
    raise typer.Exit(code=result.record.exit_code or 0)


@app.command()
def attach(run_id: str) -> None:
    """Re-open the dashboard for an existing run (view-only)."""
    from primejob.tui import attach_dashboard

    code = attach_dashboard(run_id)
    raise typer.Exit(code=code)


@runs_app.command("list")
def runs_list(
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """List recent local runs."""
    from primejob.state import list_runs

    records = list_runs(limit=limit)
    if not records:
        console.print("[dim]No runs yet.[/dim]")
        return
    table = Table(title="Recent runs")
    table.add_column("run_id", style="cyan")
    table.add_column("script")
    table.add_column("gpu")
    table.add_column("status")
    table.add_column("cost", justify="right")
    table.add_column("exit", justify="right")
    for r in records:
        cost = f"${r.total_cost:.4f}" if r.total_cost else "-"
        exit_code = str(r.exit_code) if r.exit_code is not None else "-"
        table.add_row(
            r.run_id,
            f"{r.script} {' '.join(r.args)}".strip()[:40],
            f"{r.gpu_type}×{r.gpu_count}",
            r.status,
            cost,
            exit_code,
        )
    console.print(table)


@app.command()
def status(run_id: str) -> None:
    """Show status of a run (local + remote)."""
    from primejob.backend.pods import get_status as remote_status
    from primejob.state import load_run

    record = load_run(run_id)
    console.print(f"[bold]run_id:[/bold] {record.run_id}")
    console.print(f"  pod_id: {record.pod_id}")
    console.print(f"  gpu:    {record.gpu_type} ×{record.gpu_count} @ {record.provider} ({record.country})")
    console.print(f"  script: {record.script} {' '.join(record.args)}")
    console.print(f"  status: {record.status}  exit={record.exit_code}  cost=${record.total_cost or 0:.4f}")
    console.print(f"  logs:   {record.log_path}")

    if record.pod_id and record.status == "running":
        try:
            live = remote_status(get_client(), record.pod_id)
            console.print(
                f"  [dim]live: status={live.status} install={live.installation_progress}% "
                f"rate=${live.cost_per_hr or 0:.4f}/h[/dim]"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"  [yellow]live status unavailable: {e}[/yellow]")


@app.command()
def logs(run_id: str) -> None:
    """Print the saved log file for a run."""
    from primejob.state import load_run

    record = load_run(run_id)
    if not record.log_path.exists():
        console.print(f"[yellow]No log file at {record.log_path}[/yellow]")
        raise typer.Exit(code=1)
    sys.stdout.write(record.log_path.read_text())


@app.command()
def terminate(run_id: str) -> None:
    """Force-terminate the pod for a run."""
    from primejob.backend.pods import terminate as kill_pod
    from primejob.state import load_run

    record = load_run(run_id)
    if not record.pod_id:
        console.print("[yellow]No pod_id recorded for this run.[/yellow]")
        raise typer.Exit(code=1)
    kill_pod(get_client(), record.pod_id)
    record.status = "terminated"
    if record.ended_at is None:
        import time as _t
        record.ended_at = _t.time()
    record.save()
    console.print(f"[green]Terminated pod {record.pod_id}.[/green]")


if __name__ == "__main__":
    app()
