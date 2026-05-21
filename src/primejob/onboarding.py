"""First-run onboarding: Prime login + SSH key path + Prime ssh_keys API."""
from __future__ import annotations

import socket
import subprocess
import time
import uuid
from pathlib import Path

import typer
from prime_cli.api.client import APIClient
from rich.console import Console

from primejob.auth import (
    SshKeyStatus,
    check_auth,
    check_ssh_key,
    discover_ssh_keys,
    list_registered_ssh_keys,
    matching_ssh_key_record,
    register_ssh_key,
    resolve_ssh_key_path,
    set_ssh_key_primary,
    _local_public_key_text,
)
from primejob.backend.pods import PodSpec, create_pod, terminate, wait_for_running
from primejob.backend.ssh import parse_ssh_endpoint, wait_for_ssh_connect
from primejob.auth import _fingerprint_sha256, _load_private_key
from primejob.config import load_project_config
from primejob.pricing import pick_cheapest


def run_prime_login(bin_path: str) -> int:
    return subprocess.run([bin_path, "login"]).returncode


def ensure_logged_in(console: Console, *, prime_bin: str) -> bool:
    auth = check_auth()
    if auth.client_ok:
        return True
    console.print("[yellow]No working Prime credentials yet — launching `prime login`…[/yellow]")
    code = run_prime_login(prime_bin)
    if code != 0:
        console.print("[red]`prime login` failed.[/red]")
        return False
    auth2 = check_auth()
    if not auth2.client_ok:
        console.print("[red]Auth still failing after login:[/red]", auth2.error or "")
        return False
    return True


def _pick_ssh_private_key(console: Console, *, assume_yes: bool, interactive: bool) -> Path | None:
    """Resolve which private key file to use (updates prime CLI config when needed)."""
    from prime_cli.core.config import Config

    cfg = Config()
    candidates = discover_ssh_keys()

    def loadable(path: Path) -> bool:
        try:
            _local_public_key_text(path)
            return True
        except Exception:  # noqa: BLE001
            return False

    resolved = resolve_ssh_key_path()
    if resolved is not None and loadable(resolved):
        return resolved

    working = [p for p in candidates if loadable(p)]
    if not working:
        console.print(
            "[red]No usable SSH private key found under ~/.ssh "
            "(tried id_ed25519, id_rsa, id_ecdsa).[/red]\n"
            "Generate one with:\n"
            "  [bold]ssh-keygen -t ed25519 -C \"primejob\"[/bold]\n"
            "Then re-run [bold]primejob login[/bold]."
        )
        return None

    chosen: Path | None = None
    if resolved is not None and resolved in working:
        chosen = resolved
    elif len(working) == 1:
        chosen = working[0]
    elif assume_yes:
        chosen = working[0]
    elif interactive:
        console.print("[bold]Multiple SSH keys found; pick one for Prime pods:[/bold]")
        for i, p in enumerate(working, start=1):
            console.print(f"  [{i}] {p}")
        pick = typer.prompt(
            "Choice",
            type=int,
            default=1,
            show_default=True,
        )
        if pick < 1 or pick > len(working):
            console.print("[red]Invalid choice.[/red]")
            return None
        chosen = working[pick - 1]
    else:
        chosen = working[0]

    assert chosen is not None
    cfg.set_ssh_key_path(str(chosen))
    console.print(f"[green]Saved SSH key path[/green] → [cyan]{chosen}[/cyan]")
    return chosen


def ensure_ssh_key_ready(
    client: APIClient,
    console: Console,
    *,
    assume_yes: bool = False,
    interactive: bool = True,
) -> SshKeyStatus:
    """Pick a local key, persist ssh_key_path, upload to Prime if missing, optionally set primary."""
    key_path = _pick_ssh_private_key(console, assume_yes=assume_yes, interactive=interactive)
    if key_path is None:
        return check_ssh_key(None)

    pub_text, normalized = _local_public_key_text(key_path)
    fingerprint = _fingerprint_sha256(_load_private_key(key_path))
    keys = list_registered_ssh_keys(client)
    existing = matching_ssh_key_record(keys, normalized, fingerprint)

    if existing is None:
        name_base = socket.gethostname().split(".")[0] or "machine"
        default_name = f"primejob-{name_base}"[:80]
        if interactive and not assume_yes:
            key_name = typer.prompt("Name for this key in Prime", default=default_name)
        else:
            key_name = default_name
        console.print(f"[dim]Uploading public key to Prime as '{key_name}'…[/dim]")
        try:
            register_ssh_key(client, name=key_name, public_key_line=pub_text)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Could not upload SSH key:[/red] {e}")
            return check_ssh_key(client)

    keys = list_registered_ssh_keys(client)
    match = matching_ssh_key_record(keys, normalized, fingerprint)
    if match and not match.get("isPrimary"):
        key_id = match.get("id")
        if key_id:
            make_primary = assume_yes
            if not make_primary and interactive:
                make_primary = typer.confirm(
                    "Set this key as your primary SSH key in Prime?",
                    default=False,
                )
            if make_primary:
                try:
                    set_ssh_key_primary(client, str(key_id))
                    console.print("[green]Marked SSH key as primary in Prime.[/green]")
                except Exception as e:  # noqa: BLE001
                    console.print(f"[yellow]Could not set primary key (non-fatal):[/yellow] {e}")

    return check_ssh_key(client)


def run_ssh_smoke_test(client: APIClient, console: Console, *, assume_yes: bool) -> None:
    """Provision a cheapest CPU pod, wait for SSH, then terminate (costs a small amount)."""
    cfg = load_project_config()
    if not assume_yes:
        if not typer.confirm(
            "Provision the cheapest CPU pod to verify SSH (~small cost), then terminate?",
            default=False,
        ):
            console.print("[dim]Skipping smoke test.[/dim]")
            return

    pod_id: str | None = None
    try:
        option = pick_cheapest(client, gpu_type="CPU", gpu_count=1)
        tag = uuid.uuid4().hex[:10]
        spec = PodSpec(name=f"primejob-login-smoke-{tag}", gpu_option=option)
        pod = create_pod(client, spec)
        pod_id = pod.id
        console.print(f"[dim]Smoke pod {pod_id} ({option.provider}) provisioning…[/dim]")

        def on_progress(status) -> None:
            st = (status.status or "?").lower()
            pct = status.installation_progress
            console.print(
                f"[dim]  pod={st} install={pct}% "
                f"rate=${status.cost_per_hr or 0:.4f}/h[/dim]"
            )

        pod_status = wait_for_running(client, pod_id, on_progress=on_progress)
        ssh_ep = parse_ssh_endpoint(pod_status.ssh_connection)
        console.print(
            f"[dim]Pod reachable; connecting SSH to {ssh_ep.user}@{ssh_ep.host}:{ssh_ep.port}…[/dim]"
        )

        from primejob.run import SSH_POST_READY_SLEEP_S

        if SSH_POST_READY_SLEEP_S > 0:
            console.print(
                f"[dim]  Waiting {SSH_POST_READY_SLEEP_S:.0f}s before first SSH attempt…[/dim]"
            )
            time.sleep(SSH_POST_READY_SLEEP_S)

        pod_ready_mono = time.monotonic()
        def on_retry(attempt: int, total: int, delay_s: float, detail: str) -> None:
            console.print(
                f"[dim]  [{detail}] SSH attempt {attempt}/{total}, "
                f"retry in {delay_s:.0f}s[/dim]"
            )

        client_ssh = wait_for_ssh_connect(
            ssh_ep,
            max_wait_s=float(cfg.ssh_max_wait),
            retry_delay_s=float(cfg.ssh_retry_delay),
            pod_ready_monotonic=pod_ready_mono,
            on_retry=on_retry,
        )
        client_ssh.close()
        console.print("[green]SSH smoke test succeeded.[/green]")
    finally:
        if pod_id:
            console.print(f"[dim]Terminating smoke pod {pod_id}…[/dim]")
            terminate(client, pod_id)


def run_full_login(
    *,
    prime_bin: str,
    console: Console,
    assume_yes: bool,
    smoke_test: bool,
) -> int:
    """Interactive onboarding used by `primejob login`. Returns shell exit code."""
    if not ensure_logged_in(console, prime_bin=prime_bin):
        return 1
    client = APIClient()

    mode_assume_yes = assume_yes
    mode_interactive = not assume_yes

    status = ensure_ssh_key_ready(
        client,
        console,
        assume_yes=mode_assume_yes,
        interactive=mode_interactive,
    )
    if not status.ok:
        console.print(f"[red]SSH setup incomplete:[/red] {status.error or status}")
        return 1

    console.print("[bold green]SSH key is configured and registered in Prime.[/bold green]")
    if smoke_test:
        run_ssh_smoke_test(client, console, assume_yes=assume_yes)

    console.print("[dim]Running primejob doctor…[/dim]")
    from primejob.cli import run_doctor

    return run_doctor()
