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

Kindergarten albums often contain 50–100 classroom group photos where your child appears in only a subset. The `faces` subpackage (`src/kinderpedia_backup/faces/`) scans backups created by **either** the Browser Bookmarklet or `kinderpedia-backup sync` entirely offline (`SCRFD-10GF` + `ArcFace ResNet-50` ONNX via `onnxruntime` or `opencv-python-headless` `cv2.dnn`) and organizes matching photos per child using **zero-extra-space hardlinks**:

1. Install facial-recognition extras:
   ```bash
   pip install -e "./python[faces]"
   ```
2. *(Optional)* Add extra reference portraits per child or classmate in `References/<Person>/` (`<Child>/avatar.png` is enrolled automatically):
   ```text
   ~/kinderpedia_bk/References/
   ├── Alex/
   │   ├── 2024_portrait.jpg
   │   └── 2026_portrait.jpg
   └── Maya/
       └── 2026_portrait.jpg
   ```
3. Run the incremental face scan, launch the local active-learning Web UI, and materialize flat `ByChild/<Person>/` folders or Kinderpedia CDN URL manifests:
   ```bash
   # Scan photos, join Metadata/ provenance & CDN URLs, and build per-year clusters:
   kinderpedia-backup faces scan --input ~/kinderpedia_bk

   # Launch the interactive batch-trainer & classroom studio at http://127.0.0.1:8765/:
   kinderpedia-backup faces ui --input ~/kinderpedia_bk --port 8765

   # Materialize zero-copy hardlinks into ~/kinderpedia_bk/ByChild/<Person>/:
   kinderpedia-backup faces materialize --input ~/kinderpedia_bk

   # Export Kinderpedia CDN URLs (.txt / .csv / .json) for a single child:
   kinderpedia-backup faces export-urls --input ~/kinderpedia_bk --person "Alex" --format txt -o alex_urls.txt
   ```
   - 👉 Full architecture & design doc: **[`../docs/facial-recognition-design.md`](../docs/facial-recognition-design.md)**
   - 👉 Engineering & operations handover guide: **[`../docs/faces-handover.md`](../docs/faces-handover.md)**

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
