# primejob

![primejob](docs/assets/demo.gif)

**Run your local training script on a remote GPU with one command.**

```bash
uv run train.py            # before — your laptop
primejob run train.py --gpu H100   # after — a Prime Intellect pod
```

Same `train.py`, same `uv` workflow, same stdout streamed back to your terminal. primejob picks the cheapest available GPU, uploads only what your script actually imports, attaches a reusable dataset disk, streams logs, downloads `outputs/` when done, and terminates the pod.

---

## Quick start (≈ 60 seconds)

```bash
# 1. Install
uv add git+https://github.com/AleksanderObuchowski/primejob.git

# 2. One-time auth + SSH setup (interactive)
primejob login
primejob doctor       # confirms auth, SSH key, SDK wiring

# 3. Run training
primejob run train.py --gpu H100 --yes
```

That's it. Anything after the script name is forwarded to your script:

```bash
primejob run train.py --gpu H100 -- --epochs 10 --lr 3e-4
```

`PRIME_API_KEY` is read from `.env` in cwd or from `~/.prime/config.json` (populated by `prime login` / `primejob login`).

> **Tip:** add `--setup-ssh` to `primejob run` to auto-register and promote your SSH key non-interactively the first time.

---

## Configure

Add to your project's `pyproject.toml` — most projects only need the first three keys:

```toml
[tool.primejob]
dataset_disk = "my-project-data"          # persistent disk reused across runs
forward_env  = ["HF_TOKEN", "WANDB_API_KEY"]
default_gpu  = "H200"                     # short alias resolved to H200_141GB
```

<details>
<summary><strong>Full configuration reference</strong></summary>

```toml
[tool.primejob]
# --- Compute ---
default_gpu     = "H200"                     # short alias — see GPU aliases
default_country = "US"                       # optional; biases pod placement
default_disk_size = 50                       # GB, used when creating the disk fresh
exclude_providers = ["massedcompute"]        # skip flaky providers globally

# --- Dataset ---
dataset_disk = "my-project-data"             # persistent disk reused across runs

# --- Packaging ---
include = ["data/train.jsonl", "configs/*.yaml"]   # extra files to ship (see Packaging)

# --- Environment / dependencies ---
forward_env   = ["HF_TOKEN", "WANDB_API_KEY"]
uv_extras     = ["training"]                 # passed to uv sync/run as --extra
uv_groups     = ["train"]                    # passed to uv sync/run as --group
uv_all_extras = false                        # passed to uv sync/run as --all-extras

# --- Outputs ---
download_outputs = true                      # set false to skip outputs/ download
download_include = ["outputs/**/best/**", "outputs/**/*.json"]
download_exclude = ["outputs/**/checkpoint-*/*.pt"]

# --- SSH tuning ---
ssh_max_wait     = 300                       # connect budget (seconds)
ssh_retry_delay  = 5                         # delay between SSH retries (seconds)
ssh_auth_timeout = 90                        # fallback to next provider after auth stalls
```

`bundle_paths` from older configs is still accepted as an alias for `include` (the first run prints a one-time rename hint).

</details>

---

## Everyday commands

```bash
# Datasets — one-time upload, reused by every run on the same disk
primejob dataset push ./data
primejob dataset list

# Training
primejob run train.py --gpu H100 --yes        # auto-picks cheapest matching GPU
primejob package train.py --dry-run           # preview what would be uploaded

# Inspect / control
primejob runs list
primejob status <run_id>
primejob logs   <run_id>
primejob attach <run_id>                      # re-open dashboard (view-only)
primejob terminate <run_id>                   # safety net if a run wedged

# Cost hygiene — find pods still billing because a manifest got stale
primejob runs list --check-remote
primejob runs reconcile
primejob runs reconcile --terminate-stale
```

The dashboard opens automatically in a TTY and falls back to plain streaming in CI/pipes. Pass `--plain` to force plain mode.

---

## Examples

Complete projects under [`examples/`](examples/):

- **`hf-sentiment-json`** — Hugging Face `Trainer` sentiment classification from `data/train.json`.
- **`image-folder-torch`** — Torch image-folder classification (`data/image_folder/<class>/*.ppm`).
- **`unsloth-sft`** — Unsloth + TRL SFT on a tiny in-memory dataset (GPU/CUDA smoke test).

Each example is self-contained with its own `pyproject.toml`, `uv.lock`, and README.

---

## What `primejob run` actually does

1. Walks `pyproject.toml` + `uv.lock`, validates `forward_env`.
2. Calls Prime's availability API and picks the cheapest offering matching `--gpu` / `--count` / `--country`.
3. Asks once for cost confirmation (skip with `--yes`).
4. Creates the persistent disk if missing, in the cheapest region matching country.
5. Handles the dataset according to `--data-mode` (attach by default).
6. Builds the src tarball from your entrypoint's AST closure + always-include manifest + `include` patterns; uploads via SFTP with progress.
7. Runs `uv sync` then `uv run python -u <script>` over a streaming SSH channel — stdout/stderr to your terminal AND to `~/.primejob/runs/<run_id>/log.txt`.
8. Background status bar every 30s: `[run_id] elapsed=12m34s rate=$2.43/h spent=$0.51`.
9. On exit (success, failure, or `Ctrl+C`): downloads `outputs/`, terminates the pod, waits for disk detach, writes a run manifest.

Your script sees `PRIMEJOB_DATASET_PATH` pointing at the dataset (any mode).

---

## Advanced

<details>
<summary><strong>Dataset modes (<code>--data-mode</code>)</strong></summary>

```bash
# Default — Prime attaches the persistent disk directly to the training pod.
primejob run train.py --disk my-project-data --data-mode attach

# Copy the dataset off the persistent disk before training, then release the disk
# so other jobs can use the same source (best for parallel experiments).
primejob run train.py --disk my-project-data --data-mode stage

# No persistent disk — for providers/regions where disk create fails, or HF Hub-only jobs.
primejob run train.py --gpu H100 --data-mode none --yes

# Bundle local data into the src tarball (even if gitignored).
primejob run train.py --gpu H100 --data-mode local --include data/train.jsonl --yes
```

- **`attach`** — fastest for one job. Prime currently treats the disk as exclusive, so concurrent runs on the same disk can fail with `Disk ... is already used`. primejob filters GPU availability by the disk and waits for detach after termination.
- **`stage`** — best for parallel experiments. A short-lived helper pod copies the dataset from the persistent disk to `.primejob/staged/<run_id>/`, then the training pod starts without attaching the disk. Pass `--data-subdir NAME` for large disks to stage only the needed subdirectory.
- **`none`** — skips persistent disks entirely; ignores `[tool.primejob].dataset_disk` and does not set `PRIMEJOB_DATASET_PATH`.
- **`local`** — bundles `--include` patterns into the uploaded src tarball and exposes them via `PRIMEJOB_DATASET_PATH` (`/tmp/primejob/work/...`). `--include-data` is a deprecated alias.

</details>

<details>
<summary><strong>Packaging — what gets uploaded</strong></summary>

primejob figures out what to ship by reading your entrypoint, not by uploading the working directory. Each `primejob run` (and `primejob package`) builds a `PackagePlan` from three sources:

1. **AST import closure.** Static analysis of `train.py` walks every local `import`, follows them recursively, and adds the files behind them. Third-party packages and the stdlib are skipped (they come from `uv sync` on the pod). Both flat layouts (`src/data.py`) and src-layouts (`src/<pkg>/...`) work without extra config.
2. **Always-include manifest.** `pyproject.toml`, `uv.lock`, `.python-version`, and top-level `README*` / `LICENSE*` files are added so `uv sync` has what it needs.
3. **Explicit `include` patterns.** Anything matching `[tool.primejob].include` (or `--include`/`-i` on the CLI). Globs (`configs/*.yaml`), directory shorthand (`data/`), and double-star (`runs/**/manifest.json`) all work. **This is where data files belong.**

A `DEFAULT_EXCLUDES` safety belt prunes `.git/`, `.venv/`, `.uv-cache/`, `node_modules/`, `__pycache__/`, `outputs/`, `.env`, and similar caches before the tarball.

Dynamic imports (`importlib.import_module`, `__import__`) cannot be resolved statically. primejob reports them and, in a TTY, asks once whether to add them to your `include` list; in `--yes` / non-interactive mode they become a warning and the run continues.

If the tarball ends up above 100 MB, primejob prints the five largest files so you can spot the runaway directory.

**Preview before any pod is created:**

```bash
primejob package train.py --dry-run
# Packaging plan for train.py
#   Python imports (AST closure) (3)
#     train.py
#     src/__init__.py
#     src/data.py
#   Always included (2)
#     pyproject.toml
#     uv.lock
#   Explicit include (1)
#     data/train.jsonl
#   Total: 6 files, 0.2 MB (uncompressed)
```

Drop `--dry-run` to write the tarball to disk (`./primejob-package.tar.gz` by default, or `-o path.tar.gz`). Add `-i pattern` to test extra patterns without editing `pyproject.toml`.

</details>

<details>
<summary><strong>Pod lifecycle &amp; safety</strong></summary>

`primejob run` owns the pod for the lifetime of the command. A detached local **watchdog** terminates the pod if the foreground process dies (`kill -9`, terminal closed, parent harness exit) while the machine is still up.

- **Graceful exit** (`Ctrl+C`, errors, normal finish): cleanup terminates the pod and finalizes `~/.primejob/runs/<run_id>/manifest.json`.
- **Abrupt local death**: the watchdog detects a dead parent PID or stale lease heartbeat and terminates the pod.
- **Recovery**: `primejob runs list --check-remote`, `primejob runs reconcile`, and `primejob terminate <run_id>` surface and fix stale `running` manifests.

A host crash or OOM that kills both the CLI and watchdog cannot be covered locally; use `primejob runs reconcile` when you return.

**Manual repro (`kill -9`):**

```bash
primejob run train.py --gpu <small_gpu> --yes --plain
# After "SSH connected" / pod_id is in manifest:
kill -9 <primejob_pid>
# Within ~5s the watchdog should terminate the pod; manifest status → terminated.
prime pods list
primejob status <run_id>
```

</details>

<details>
<summary><strong>SSH troubleshooting</strong></summary>

Prime pods receive your account's **primary** SSH public key — registration alone is not always enough. `primejob doctor` reports whether your local key is registered and marked primary. `primejob run` (and smoke tests) send your registered key's `sshKeyId` in the pod create payload so Prime injects it promptly on slow providers.

Some providers expose SSH while `authorized_keys` is still empty or never populated by Prime's backend. If you see repeated `SSH [auth_propagation]` retries for more than ~1 minute, it is usually **not** a local config issue — Prime may not have injected your key into that provider's VM.

**Workarounds:**

- Confirm primary status: `primejob doctor` or `primejob login --yes`
- Exclude broken providers on the CLI: `primejob run --skip-provider massedcompute --skip-provider nebius`
- Or persist it in `pyproject.toml`:

  ```toml
  [tool.primejob]
  exclude_providers = ["massedcompute", "nebius", "crusoecloud"]
  ```

- Try a different `--country` to land on another provider
- Use `primejob run --setup-ssh` before provisioning to register and promote your key non-interactively

Tune retry budgets via `[tool.primejob].ssh_max_wait`, `ssh_retry_delay`, `ssh_auth_timeout` (see the configuration reference).

**About `primejob login`:** runs `prime login` when credentials are missing, picks a working key under `~/.ssh/` (`id_ed25519`, then `id_rsa`, then `id_ecdsa`), saves `ssh_key_path` in Prime CLI config, uploads the public key to your Prime account via the API when needed, optionally promotes it to primary, then runs `primejob doctor`. Flags: `--yes` / `-y` for non-interactive defaults; `--smoke-test` provisions the cheapest CPU pod, waits for SSH, and terminates it (small cost) to validate end-to-end access.

</details>

<details>
<summary><strong>Dashboard keybindings (TTY mode)</strong></summary>

| Key   | Action                                                |
|-------|-------------------------------------------------------|
| `q`   | quit                                                  |
| `^C`  | terminate run (with confirm)                          |
| `/`   | search log                                            |
| `p`   | pause auto-scroll                                     |
| `e`   | edit log in `$EDITOR`                                 |
| `o`   | open outputs                                          |
| `t`   | cycle theme                                           |
| `g`   | toggle GPU panel                                      |
| `s`   | show `pod_id`                                         |
| `k`   | show `run_id`                                         |
| `?`   | help                                                  |

</details>

<details>
<summary><strong>GPU type aliases</strong></summary>

| Short    | Full              |
|----------|-------------------|
| H100     | H100_80GB         |
| H200     | H200_141GB        |
| A100     | A100_80GB         |
| B200     | B200_180GB        |
| RTX4090  | RTX4090_24GB      |
| CPU      | CPU_NODE          |

Full list: `primejob gpus list`.

</details>

<details>
<summary><strong>Where state lives</strong></summary>

- `~/.primejob/runs/<run_id>/manifest.json` — per-run record (gpu, cost, exit code).
- `~/.primejob/runs/<run_id>/log.txt` — full captured stdout/stderr.
- `./outputs/<run_id>/` — files your script wrote under `outputs/`, downloaded after the run.

</details>

---

**For LLM assistants / coding agents:** see [`llms.txt`](llms.txt) for a single-file overview of primejob's CLI surface, configuration keys, runtime contract, and behavioral guarantees — written to be pasted into an agent context.
