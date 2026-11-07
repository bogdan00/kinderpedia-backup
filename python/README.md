# 🐍 Kinderpedia Backup — Python CLI & Face Organizer

Modular, resumable Python CLI for backing up full-resolution Kinderpedia media (`sync`) and organizing classroom group albums per child using local facial recognition (`faces`).

---

## Requirements

- **Python 3.10+**
- **`ffmpeg`** *(optional, needed only for HLS-streamed Vimeo videos during `sync`)*:
  - **macOS**: `brew install ffmpeg`
  - **Windows**: `winget install Gyan.FFmpeg`
  - **Ubuntu / Debian**: `sudo apt install ffmpeg`

---

## Installation & Quick Start

### Option A: Zero-setup run with [`uv`](https://docs.astral.sh/uv/)

```bash
# From repository root:
uv run python/kinderpedia.py sync
```

### Option B: Install into a virtual environment or `pipx`

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ./python

kinderpedia-backup sync
```

---

## 1. Cloud Backup (`sync`)

On first run, `kinderpedia-backup sync` prompts interactively for your Kinderpedia parent email and password (plus 2FA code if enabled), discovers all enrolled children, and downloads albums (`<Child>/Albums/<Year>/<Month>/<YYYY>-<MM>-<DD> <Title>/`), day-matched classroom MP4 videos (co-located directly inside their day album folder), and documents (`<Child>/Documents/`) to `~/kinderpedia_bk` (or `--output`) using the **exact same folder & `Metadata/` layout as the Browser Bookmarklet** (zero `.json` sidecars in media folders, seamlessly cross-resumable between Bookmarklet and Python CLI).

```bash
# Dry-run inventory (list children, albums, photo counts, videos, and documents):
kinderpedia-backup sync --discover-only

# Back up to an external SSD or cloud-synced folder:
kinderpedia-backup sync --output "/Volumes/FamilySSD/Kinderpedia"

# Incremental sync next month (fetches only newly posted items):
kinderpedia-backup sync --refresh-index

# Filter to one child, photos only:
kinderpedia-backup sync --child "Alex" --skip-videos --skip-documents

# Non-interactive password via 1Password CLI / Keychain (never visible in `ps aux`):
op read "op://Private/Kinderpedia/password" | kinderpedia-backup sync --email parent@example.com --password-stdin

# Remove saved session tokens from disk:
kinderpedia-backup sync --logout
```

---

## 2. Facial Recognition & Per-Child Organization (`faces`)

Kindergarten albums often contain 50–100 classroom group photos where your child appears in only a subset. The `faces` subpackage (`src/kinderpedia_backup/faces/`) is architected to scan backups created by **either** the Browser Bookmarklet or `kinderpedia-backup sync` and organize matching photos per child using **zero-extra-space hardlinks**:

1. Create a `References/` folder with 3–5 clear portraits per child:
   ```text
   ~/kinderpedia_bk/References/
   ├── Alex/
   │   ├── portrait_1.jpg
   │   └── portrait_2.jpg
   └── Maya/
       ├── portrait_1.jpg
       └── portrait_2.jpg
   ```
2. Preview discovered albums and reference profiles:
   ```bash
   kinderpedia-backup faces --input ~/kinderpedia_bk --dry-run
   ```
3. **Extending with an ML Recognizer**:
   - `kinderpedia_backup.faces` already implements reference enrollment (`enroll_references`), incremental scan caching (`face_scan_cache.json` keyed by `model_id`, file size, and `mtime_ns`), cosine similarity matching, and `hardlink`/`symlink`/`copy` materialization into `<input>/ByChild/<Child Name>/`.
   - To add a concrete local ML backend (e.g. InsightFace ONNX or Apple Vision), implement the `FaceRecognizer` protocol in `src/kinderpedia_backup/faces/recognizer.py` and install optional ML extras with `pip install "./python[faces]"`.
   - 👉 Full architecture & ML backend design: **[`../docs/facial-recognition-design.md`](../docs/facial-recognition-design.md)**

---

## Security & Development

- **Private Session Storage**: Auth tokens live in `~/.config/kinderpedia-backup/session.json` (`0700` dir / `0600` file on POSIX; `%LOCALAPPDATA%\kinderpedia-backup` on Windows), never inside `--output`.
- **Redacted Diagnostics**: `redact_secrets()` scrubs tokens, `Bearer` headers, and signed S3 query parameters from all logs and exceptions.

### Run Checks Locally

```bash
cd python
pip install -e ".[dev]"
ruff check src tests kinderpedia.py
mypy src tests kinderpedia.py
pytest -v
```
