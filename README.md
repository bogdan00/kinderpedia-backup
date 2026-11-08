# ☁️ Kinderpedia Cloud Backup & Organizer

Download and preserve all of your children's full-resolution Kinderpedia photo albums, classroom Vimeo videos (MP4), and school documents on your own computer — with two ways to run and a local post-processing pipeline for per-child facial recognition.

---

## Choose How You Want to Back Up

| Feature | 🌐 [1-Click Browser Bookmarklet](./bookmarklet/) | 🐍 [Python CLI & Organizer](./python/) |
| :--- | :--- | :--- |
| **Setup** | **Zero install** — drag a bookmark in Chrome / Edge / Brave | Python 3.10 – 3.14 (`uv run --python 3.14 python/kinderpedia.py` or `pip install "./python[faces]"`) |
| **Best for** | Parents wanting an interactive 1-click backup in the browser | Large archives (20–50+ GiB), scheduled/headless syncs, Safari/Firefox users |
| **Auth** | Reuses your active `app.kinderpedia.co` browser tab | Interactive prompt, `--password-stdin` (1Password/Keychain), or `0600` `.env` |
| **Videos** | Pure-JS HLS-to-MP4 transmuxing (`mux.js`) | Progressive MP4 + lossless `ffmpeg` HLS remuxing |
| **Resume** | Instant resume via `Metadata/` manifests | Instant cross-resume with Bookmarklet (`Metadata/` manifests) |
| **Face Sorting** | 100% identical output layout | Built-in `faces` subcommand architecture (local-only per-child sorting) |

---

## Quick Start

### Option 1: 1-Click Browser Bookmarklet (No Install)

1. Open **[https://bogdan00.github.io/kinderpedia-backup/](https://bogdan00.github.io/kinderpedia-backup/)** in Chrome, Edge, or Brave.
2. Drag the **💾 Kinderpedia Backup** button to your Bookmarks Bar (`Cmd+Shift+B` on Mac / `Ctrl+Shift+B` on Windows).
3. Sign in to **[app.kinderpedia.co](https://app.kinderpedia.co)**, click the bookmark, pick a destination folder, and click **Start Backup**.

👉 Full browser & helper docs: **[`bookmarklet/README.md`](./bookmarklet/README.md)**

---

### Option 2: Python CLI Backend (Unattended Sync & Post-Processing)

**Requirements:** Python 3.10 – 3.14 (`python/.python-version` pins 3.14). Facial recognition (`faces`) runs on `onnxruntime` (CoreML on macOS, CUDA when available, else CPU), which has no Python 3.15 wheels yet.

Run directly with [`uv`](https://docs.astral.sh/uv/) (zero manual virtualenv setup):

```bash
uv run --python 3.14 python/kinderpedia.py sync
```

Or install into a virtual environment:

```bash
python3.14 -m venv .venv && .venv/bin/pip install "./python[faces]"
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Full backup (prompts securely for email, password, and 2FA if enabled):
kinderpedia-backup sync --output ~/kinderpedia_bk

# Incremental sync later (plain `sync` only lists what changed and fetches newly posted items):
kinderpedia-backup sync

# Force a full re-listing of every album (slow; only if something looks out of date):
kinderpedia-backup sync --refresh-index
```

👉 Full CLI, security, and facial-recognition docs: **[`python/README.md`](./python/README.md)** · **[`docs/facial-recognition-design.md`](./docs/facial-recognition-design.md)** · **[`docs/faces-handover.md`](./docs/faces-handover.md)**

---

## Shared Backup Folder Structure

Both the **Browser Bookmarklet** and the **Python CLI** produce the **exact same folder & `Metadata/` format** — using short child names, `<Year>/<Month>/<YYYY>-<MM>-<DD> <Title>` album folders, day-matched classroom videos directly inside their corresponding album folders, and zero JSON clutter inside media folders:

```text
Your Backup Folder/
├── Alex/                            # Short first-name folder (auto-disambiguated if needed)
│   ├── avatar.png
│   ├── Albums/
│   │   └── 2025/
│   │       └── 06/
│   │           ├── 2025-06-12 Summer Festival/
│   │           │   ├── 001_987654.jpg
│   │           │   ├── 002_987655.jpg
│   │           │   └── Classroom Song.mp4   # Videos auto-matched to their day album!
│   │           └── 2025-06-18 Recital/      # Standalone dated folder on video-only days
│   │               └── Recital.mp4
│   └── Documents/
│       └── Weekly Menu.pdf
├── Metadata/                        # All index, progress, & per-album JSON manifests live here
│   ├── backup_summary.json
│   └── Alex/
│       ├── albums_index.json
│       ├── videos_index.json
│       ├── documents_index.json
│       ├── backup_progress.json
│       ├── Albums/
│       │   └── 2025-06-12 Summer Festival.json
│       └── Videos/
└── ByChild/                         # Created by `kinderpedia-backup faces` (zero-copy hardlinks)
    ├── Alex/
    └── Maya/
```

---

## Repository Layout

```text
kinderpedia-backup/
├── bookmarklet/                 # 🌐 1-Click Browser Bookmarklet + GitHub Pages UI
│   ├── index.html               # Landing & install page
│   ├── kinderpedia-backup.js    # Browser backup engine (File System Access API + Shadow DOM)
│   ├── mux.min.js               # In-browser HLS-to-MP4 remuxer
│   ├── worker.js                # Cloudflare Worker CORS relay for Vimeo player configs
│   └── serve.js                 # Optional local Node.js dev & ffmpeg streaming server
│
├── python/                      # 🐍 Python CLI Package (`kinderpedia-backup`)
│   ├── kinderpedia.py           # Zero-setup PEP 723 runner (`uv run python/kinderpedia.py`)
│   ├── pyproject.toml           # Core `sync` + optional `[faces]` & `[dev]` extras
│   ├── src/kinderpedia_backup/
│   │   ├── cli.py               # Unified CLI (`sync` and `faces` subcommands)
│   │   ├── security.py          # Secret redaction, 0700/0600 private state dir, path guards
│   │   ├── sync/                # Cloud API client, AIMD rate limiter, resumable downloader
│   │   └── faces/               # Offline facial-recognition studio & hardlink organizer
│   └── tests/                   # Unit test suite (`pytest` / `unittest`)
│
├── docs/                        # 📐 Architecture & Engineering Handover Docs
│   ├── facial-recognition-design.md
│   └── faces-handover.md
│
├── wrangler.toml                # Cloudflare Worker deployment config (`bookmarklet/worker.js`)
└── .github/workflows/
    ├── ci.yml                   # Gitleaks secret scan + Ruff + Mypy --strict + Pytest matrix
    └── pages.yml                # Publishes browser static bundle to GitHub Pages
```

---

## Privacy & Security

- **100% Local Media Transfer**: Your photos, videos, documents, and credentials never touch a third-party storage server — downloads stream directly from Kinderpedia/Vimeo CDNs to your disk.
- **Cloud-Sync Safe State**: The Python CLI stores auth tokens in an OS-private `0700`/`0600` config directory (`~/.config/kinderpedia-backup` on macOS/Linux, `%LOCALAPPDATA%\kinderpedia-backup` on Windows) rather than inside your `--output` media folder, so backing up directly into Google Drive, iCloud, or Dropbox never uploads session tokens.
- **Local-Only Facial Recognition**: The `faces` module is designed to compute face embeddings strictly offline on your own CPU/GPU using zero-extra-space filesystem hardlinks.

---

## License

MIT License — see [LICENSE](./LICENSE). Not affiliated with or endorsed by Kinderpedia.
