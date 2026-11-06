# ☁️ Kinderpedia Backup for Parents

A simple, 1-click tool for parents to download and keep all their children's kindergarten photos, videos, and school documents safely on their own computer.

👉 **Install it here:** **[https://bogdan00.github.io/kinderpedia-backup/](https://bogdan00.github.io/kinderpedia-backup/)**

---

## 🌟 Why Use This?

- 🖱️ **Zero Installation**: You don't need to install any apps or software. It runs right in your web browser.
- 📸 **Saves Everything**: Downloads full-resolution photo albums, classroom videos, and documents.
- 👶 **All Your Kids in One Place**: Automatically finds all your children enrolled on your Kinderpedia account.
- ⚡ **Fast & Lightweight**: Downloads files directly to your computer without slowing down your browser.
- 🔄 **Smart Resuming**: You can pause or stop anytime. When you run it again, it skips what you already saved and only downloads new memories.
- 💻 **Works Everywhere**: All filenames are neatly cleaned so they open smoothly on Windows PCs, Macs, USB sticks, and digital photo frames.
- 🔒 **100% Private**: Runs entirely on your computer. None of your photos, passwords, or personal details are ever shared or sent anywhere.

---

## 🚀 How to Use (3 Simple Steps)

### Step 1: Add the Bookmark
1. Open **[https://bogdan00.github.io/kinderpedia-backup/](https://bogdan00.github.io/kinderpedia-backup/)** in Google Chrome, Microsoft Edge, or Brave.
2. Make sure your browser's **Bookmarks Bar** is showing (press `Ctrl + Shift + B` on Windows, or `Cmd + Shift + B` on Mac).
3. Drag the big blue **`☁️ Kinderpedia Backup`** button up into your Bookmarks Bar.

### Step 2: Open Kinderpedia
1. Go to **[https://app.kinderpedia.co](https://app.kinderpedia.co)** and log in as you normally do.

### Step 3: Start Your Backup
1. Click your new **`☁️ Kinderpedia Backup`** bookmark.
2. A small window will appear showing your children's names.
3. Click **Select Destination Folder** and pick where you want your photos saved (like your Desktop or a USB stick).
4. Click **Start Backup** and watch your memories download!

---

## 📁 How Your Files Are Organized

Everything is saved in clean, easy-to-browse folders:

```text
📁 Your Backup Folder/
├── 📁 Emilia/
│   ├── 📁 Albums/       <-- All photo albums, sorted by date and name
│   ├── 📁 Videos/       <-- All classroom and activity videos
│   └── 📁 Documents/    <-- School reports, newsletters, and documents
└── 📁 Radu/
    ├── 📁 Albums/
    ├── 📁 Videos/
    └── 📁 Documents/
```

---

## ❓ Frequently Asked Questions

**Do I need to install any program on my computer?**  
No. It works completely inside your web browser using a simple bookmark.

**Is it safe and private?**  
Yes, 100%. The download happens directly between Kinderpedia and your computer. No external server ever sees your photos, account, or password.

**Can I save directly to an external hard drive or USB stick?**  
Yes! When you click *Select Destination Folder*, simply choose your external drive or USB stick.

**What happens if my internet disconnects?**  
No problem. Just click the bookmark again and select the same folder. It will instantly pick up right where it left off without downloading duplicates.

**Which web browsers are supported?**  
Google Chrome, Microsoft Edge, Brave, Opera, and other modern Chromium-based browsers on Windows, Mac, or Linux.

---

## 🎬 Classroom Videos (Optional Helper)

Photos and school documents download directly with zero extra setup.

Kinderpedia hosts classroom videos on Vimeo, which places restrictions on web browsers downloading videos directly. By default, the tool will safely catalog all your video details into a neat index file.

If you would like your videos to download directly onto your computer as `.mp4` files without running any software, you can use a free Cloudflare Worker:
1. Log in to [Cloudflare](https://dash.cloudflare.com/) (it's 100% free).
2. Go to **Workers & Pages** ➔ **Create application** ➔ **Create Worker**.
3. Name it `kinderpedia-video-helper` and click **Deploy**.
4. Click **Edit code**, paste the contents of [`worker.js`](worker.js), and click **Deploy**.
5. Copy your worker address (e.g. `https://kinderpedia-video-helper.<your-subdomain>.workers.dev`).
6. In your Kinderpedia Backup bookmarklet, click **⚡ Cloudflare Video Helper (Optional)**, paste your worker address, and your videos will download automatically!
