# ☁️ Kinderpedia Cloud Backup — 1-Click Browser Bookmarklet

A fast, private, zero-install browser bookmarklet that allows parents to backup their children's complete photo albums, videos, and documents directly from [Kinderpedia](https://app.kinderpedia.co) to a local computer or external drive.

🔗 **Live Bookmarklet Page**: **[https://bogdan00.github.io/kinderpedia-backup/](https://bogdan00.github.io/kinderpedia-backup/)**

---

## ✨ Features

- **Zero-Install (1-Click)**: No Python, terminal commands, or native software required. Runs directly inside your web browser.
- **Direct-to-Disk Streaming**: Uses the modern [File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_Access_API) (`showDirectoryPicker`) to stream high-resolution media straight to disk without overloading your computer's RAM.
- **Concurrent Producer-Consumer Architecture**: Discovers upcoming albums in real time while 6 parallel streaming workers download photos concurrently. Total counts and progress adjust dynamically.
- **Multi-Child Detection**: Automatically recognizes all child profiles associated with your parent account.
- **Instant Resume & Incremental Sync**: If paused, stopped, or re-run in the future, it skips already downloaded files in milliseconds and only downloads newly uploaded media.
- **Windows, FAT32 & USB Safe**: All file and directory names are cleaned automatically—stripping emojis, normalizing curly quotes, dashes, and ellipsis, and removing trailing dots/spaces that cause errors on Windows or external USB drives.
- **Pure Media Organization**: Photo, video, and document folders contain pure media files only. All JSON indexes and manifests are neatly segregated into a dedicated `Metadata/` folder.
- **Strict Data Integrity**: Every photo, video, and document is checked against its expected size. Incomplete or 0-byte files are automatically deleted and re-downloaded.
- **100% Private & Secure**: Operates entirely within your active browser session. No credentials, tokens, or photos ever touch any third-party server.

---

## 🚀 How to Use (For Parents)

### 1. Install the Bookmarklet
1. Open **[https://bogdan00.github.io/kinderpedia-backup/](https://bogdan00.github.io/kinderpedia-backup/)** in Chrome, Edge, Brave, or any Chromium-based browser.
2. Ensure your browser's **Bookmarks Bar** is visible (`Ctrl + Shift + B` on Windows, `Cmd + Shift + B` on Mac).
3. Drag the blue **`☁️ Kinderpedia Backup`** button directly into your Bookmarks Bar.

### 2. Run the Backup
1. Go to **[https://app.kinderpedia.co](https://app.kinderpedia.co)** (or `https://mykp.kinderpedia.co`) and log in.
2. Click your **`☁️ Kinderpedia Backup`** bookmark.
3. An overlay dashboard will appear showing your logged-in account and your children's profiles.
4. Click **Select Destination Folder** and choose where you want to save the backup (e.g., your Desktop, Documents, or an external USB hard drive).
5. Click **Start Backup**!

---

## 📁 Backup Folder Structure

The backup engine structures your library cleanly and predictably:

```text
[Your Chosen Backup Folder]/
├── 📁 Emilia Draghicescu/
│   ├── avatar.png
│   ├── 📁 Albums/
│   │   ├── 📁 2026-09-07_47819249_Traseu sportiv/
│   │   │   ├── 001_47819317.jpg
│   │   │   ├── 002_47819318.jpg
│   │   │   └── ...
│   │   └── 📁 2026-10-21_49893161_Masti vesele - Culori si creativitate/
│   │       ├── 001_49893201.jpg
│   │       └── ...
│   ├── 📁 Videos/
│   │   ├── 2026-09-18_48353657_Salutul de dimineata.mp4
│   │   ├── 2026-10-12_49447004_Ghici ghicitoarea mea.mp4
│   │   └── ...
│   └── 📁 Documents/
│       ├── Formular optionale.docx
│       ├── Program serbare.pdf
│       └── ...
└── 📁 Metadata/
    └── 📁 Emilia Draghicescu/
        ├── albums_index.json
        ├── videos_index.json
        ├── documents_index.json
        ├── backup_progress.json
        └── 📁 Albums/
            ├── 2026-09-07_47819249_Traseu sportiv.json
            └── ...
```

---

## 🔒 Privacy & Security

- **Where do my files go?**
  Directly from Kinderpedia's CDN servers to your local hard drive. There is no middleman server.
- **Are my credentials safe?**
  Yes. The bookmarklet runs entirely within your existing logged-in browser session. It does not ask for or store your password, nor does it transmit your session tokens anywhere.
- **Can I inspect the code?**
  Yes! The script is completely open source and un-obfuscated in this repository (`kinderpedia-backup.js`).

---

## 🌐 Browser Compatibility

| Browser | Supported | Notes |
| :--- | :---: | :--- |
| **Google Chrome** | ✅ Yes | Recommended |
| **Microsoft Edge** | ✅ Yes | Fully supported |
| **Brave** | ✅ Yes | Fully supported |
| **Opera / Vivaldi** | ✅ Yes | Fully supported |
| **Mozilla Firefox** | ⚠️ Limited | Lacks native File System Access API (`showDirectoryPicker`) |
| **Apple Safari** | ⚠️ Limited | Lacks native File System Access API (`showDirectoryPicker`) |

---

## 🎬 Note on Vimeo Videos

Kinderpedia hosts parent-uploaded videos on Vimeo with domain privacy restrictions.
- Direct MP4 files and documents download 100% in-browser without any setup.
- If Vimeo restricts in-browser cross-origin streaming for certain video formats, you can optionally run the lightweight local helper `node serve.js` on port `8080`, which uses `ffmpeg` to transmux Vimeo HLS streams directly into lossless MP4 files.

---

## 📄 License

MIT License. Feel free to use, share, and adapt for your family's memories.
