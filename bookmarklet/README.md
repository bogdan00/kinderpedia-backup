# 🌐 Kinderpedia Backup — Browser Bookmarklet

Zero-install, 1-click browser tool for parents to download all kindergarten photo albums, Vimeo classroom videos, and documents directly from an active Kinderpedia browser session.

👉 **Live Install Page:** **[https://bogdan00.github.io/kinderpedia-backup/](https://bogdan00.github.io/kinderpedia-backup/)**

*(Looking for unattended CLI sync or local facial-recognition sorting? See the [Python CLI in `../python/`](../python/README.md).)*

---

## How to Use (3 Steps)

1. **Add the Bookmark**: Open **[https://bogdan00.github.io/kinderpedia-backup/](https://bogdan00.github.io/kinderpedia-backup/)** in Chrome, Edge, or Brave and drag **`💾 Kinderpedia Backup`** onto your Bookmarks Bar (`Cmd+Shift+B` on Mac / `Ctrl+Shift+B` on Windows).
2. **Sign In to Kinderpedia**: Open **[https://app.kinderpedia.co](https://app.kinderpedia.co)** and log in normally.
3. **Start Backup**: Click the bookmark, pick a destination folder on your computer or external drive, and click **Start Backup**.

---

## Output Folder Layout

```text
📁 Your Backup Folder/
├── 📁 Alex/
│   ├── 📁 Albums/
│   │   └── 📁 2025/
│   │       └── 📁 06/
│   │           └── 📁 2025-06-12 Summer Festival/   <-- Photos + same-day classroom MP4 videos
│   └── 📁 Documents/                                <-- Newsletters, menus, and PDFs
├── 📁 Maya/
│   ├── 📁 Albums/
│   └── 📁 Documents/
└── 📁 Metadata/                                     <-- Resume indexes & manifests (shared with Python CLI)
```

---

## Developer & Self-Hosting Notes

### Files in `bookmarklet/`

- `index.html` — Install & documentation page (deployed to GitHub Pages via `.github/workflows/pages.yml`).
- `kinderpedia-backup.js` / `kinderpedia-backup.min.js` — Main browser engine using the File System Access API (`showDirectoryPicker`), Shadow DOM UI, and AIMD adaptive rate limiting.
- `mux.min.js` — Browser HLS-to-MP4 transmuxer so Vimeo HLS streams save as standard `.mp4` files without requiring native `ffmpeg`.
- `worker.js` — Stateless Cloudflare Worker (configured via root `../wrangler.toml`) that relays Vimeo embed player JSON metadata with CORS headers so browser `fetch()` calls can resolve video stream URLs.
- `serve.js` — Optional local Node.js development & fallback Vimeo streaming server (`node bookmarklet/serve.js`, listens on `http://localhost:8080`).

### GitHub Pages Deployment

`.github/workflows/pages.yml` bundles only the browser-facing static assets (`index.html`, `favicon.*`, `kinderpedia-backup*.js`, `mux.min.js`) and publishes them to the root Pages URL (`https://bogdan00.github.io/kinderpedia-backup/`). Ensure **Settings → Pages → Build and deployment → Source** is set to **GitHub Actions** in your GitHub repository settings.
