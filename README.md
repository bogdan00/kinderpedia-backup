# ☁️ Kinderpedia Cloud Backup — 1-Click Bookmarklet

A secure, private, zero-install browser bookmarklet that allows parents to backup their children's photo albums, videos, and documents directly from [Kinderpedia](https://app.kinderpedia.co) to local disk storage using modern browser APIs.

---

## ✨ Features

- **Zero-Install (1-Click)**: No command-line tools, Python, or native software required.
- **Direct-to-Disk Streaming**: Uses the standard [File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_Access_API) (`showDirectoryPicker`) to stream photos, videos, and documents directly to your local drive without loading them into browser memory.
- **Concurrent Producer-Consumer Engine**: Discovers upcoming albums in real time while 6 parallel workers download photos concurrently.
- **Windows & USB Safe**: All file and directory names are automatically cleaned (no emojis, no trailing dots/spaces, normalized quotes/dashes, and escaped DOS reserved names).
- **Strict Data Integrity**: Files are verified against expected byte counts; corrupted or incomplete files are automatically removed and re-downloaded.
- **Safe & Private**: Runs 100% locally in your browser session. No credentials or files ever touch any third-party server.

---

## 🚀 How to Use (For Parents)

1. Open the hosted page: **`https://<your-username>.github.io/<your-repo>/`**
2. Make sure your browser's **Bookmarks Bar** is visible (`Ctrl+Shift+B` or `Cmd+Shift+B`).
3. Drag the big blue **`☁️ Kinderpedia Backup`** button directly into your bookmarks bar.
4. Log into [Kinderpedia](https://app.kinderpedia.co) (or `https://mykp.kinderpedia.co`).
5. Click your new **`☁️ Kinderpedia Backup`** bookmark.
6. Pick a folder on your computer (e.g., Desktop or an external USB hard drive) and click **Start Backup**!

---

## 🛠️ Repository Contents

- `index.html`: The GitHub Pages landing page with the drag-and-drop bookmarklet installation button.
- `kinderpedia-backup.js`: The client-side backup engine injected by the bookmarklet.
- `kinderpedia-backup.min.js`: Pre-synced script distribution.
- `serve.js`: Optional local Node helper for remuxing high-resolution Vimeo videos via ffmpeg.

---

## 📄 License

MIT
