# Kaggle from an agent — verified config and notebook execution

Everything below was verified by actually running it (the
[`laya-evals-reproduction`](https://www.kaggle.com/code/gjusev/laya-evals-reproduction)
kernel was created, monitored, completed, and made public with exactly these
commands). Hand this file to any agent that needs to work with Kaggle.

## 1. Auth: token, never OAuth

Kaggle's OAuth endpoint rejects agent clients (it demands
`client_secret_basic`). Use a token: kaggle.com → Settings → API → create
token (format `KGAT_...`).

```bash
# CLI (the reliable path): plain env var, nothing installed globally
export KAGGLE_API_TOKEN=KGAT_xxx
# or one-shot with uv, no project pollution:
KAGGLE_API_TOKEN=KGAT_xxx uv run --with kaggle python -m kaggle <command>

# MCP server (optional; good for search/read, see caveats below):
claude mcp add kaggle https://www.kaggle.com/mcp --scope user \
  --transport http --header "Authorization: Bearer KGAT_xxx"
```

The token never goes in a file, a commit, or a kernel. Pass it as an env
var at invocation time only.

## 2. Which runtime to use for what (verified)

| Task | Use | Notes |
|---|---|---|
| Search datasets/competitions | MCP tools (`mcp__kaggle__search_datasets`, ...) | Work fine read-only |
| Download a dataset | `kagglehub` | `uv run --with kagglehub python -c "import kagglehub; kagglehub.dataset_download('owner/slug')"` — works tokenless for public data; cache lands in `~/.cache/kagglehub/` |
| Push/monitor/download kernels | CLI `python -m kaggle kernels ...` | Verified: push, status, output, list |
| Discover your username | `kaggle kernels list --mine` | The `author` column is the account name |

Known MCP quirks (do not burn time on them):

- `authorize` returns a malformed schema result — ignore it; search still
  works.
- `download_dataset` hands back a signed GCS URL that 403s (SignatureDoesNotMatch)
  — use kagglehub instead.
- The SSE stream escapes quotes (`"`), so grepping `status` never
  matches. If you must read SSE, parse `data:` lines with `json.loads`
  (JSON inside JSON). Direct JSON-RPC also works:

```bash
curl -s -X POST https://www.kaggle.com/mcp \
  -H "Authorization: Bearer $KAGGLE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"TOOLNAME","arguments":{"request":{...}}}}'
```

## 3. Running a notebook (kernel)

A kernel is a folder with `kernel-metadata.json` + one code file:

```json
{
  "id": "YOUR_USERNAME/slug-of-kernel",
  "title": "slug-of-kernel",
  "code_file": "script.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

- `kernel_type`: `"script"` or `"notebook"`; for notebooks `code_file` is
  the complete `.ipynb`.
- The slug ALWAYS carries the `username/` prefix.
- `machine_shape` (GPU): `NvidiaTeslaT4` (= T4 x2), `NvidiaTeslaP100`,
  `Tpu1VmV38`.

Push (creates a new version and executes it SaveAndRunAll):

```bash
KAGGLE_API_TOKEN=KGAT_xxx uv run --with kaggle \
  python -m kaggle kernels push -p <folder>
# -> "Kernel version N successfully pushed"
```

Monitor and collect:

```bash
KAGGLE_API_TOKEN=... python -m kaggle kernels status USER/SLUG
# KernelWorkerStatus.RUNNING | COMPLETE | ERROR
KAGGLE_API_TOKEN=... python -m kaggle kernels output USER/SLUG -p <dir>
# log + files into <dir> (a 'charmap' warning on Windows is cosmetic)
```

## 4. The traps (each cost real time once)

1. **GPU silently absent.** Without phone verification, the push saves the
   GPU settings but the container starts without a GPU (`nvidia-smi: command
   not found`). No error is raised. Verify the account, or stay on CPU.
2. **CPU is slow.** Kaggle CPU is 2-4 cores. A job that takes 20 minutes on
   a laptop took ~1h there. Poll every few minutes; do not assume a hang.
3. **`kernel_sources` silently dropped** when the source kernel's last
   version errored. Attach data as a dataset instead:
   `kaggle datasets create -p <folder> --dir-mode zip` (without
   `--dir-mode zip`, subfolders are ignored), then list it under
   `dataset_sources`.
4. **A kernel that dies in under 2 minutes** usually failed on the first
   cell (dependency/import error) — read the log via `kernels output`.
5. **Making a kernel public** = set `is_private: false` in the metadata and
   push again (a new version; the previous completed run stays attached to
   its version).
6. **GPU quota**: 30h/week total (T4x2 consumes it fastest). Check with the
   MCP `get_accelerator_quota` tool.

## 5. Worked example

`kaggle-kernel/` in this repository is the exact kernel that re-ran the
laya benchmark reproduction on Kaggle: metadata + a `script.py` that clones
a public GitHub repo, pip-installs it, runs the measurement, and prints the
verdict table. Pattern to copy for "run this repo's evaluation on Kaggle":

```python
import subprocess, sys
def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

run("git clone --depth 1 https://github.com/OWNER/REPO.git")
run(f"{sys.executable} -m pip install -q --no-input './REPO[extra]'")
run(f"{sys.executable} REPO/scripts/measure.py --out /kaggle/working/result.json")
```

Write outputs to `/kaggle/working/` — that is what `kernels output`
downloads.
