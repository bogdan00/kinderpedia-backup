const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { spawn, execSync } = require("child_process");

const PORT = 8080;
const DIR = __dirname;

const MIME = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".json": "application/json",
  ".css": "text/css",
  ".png": "image/png",
};

let cachedFfmpegPath = null;
function findFfmpeg() {
  if (cachedFfmpegPath) return cachedFfmpegPath;
  const candidates = [
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      cachedFfmpegPath = c;
      return c;
    }
  }
  try {
    const which = execSync("which ffmpeg", { encoding: "utf8" }).trim();
    if (which && fs.existsSync(which)) {
      cachedFfmpegPath = which;
      return which;
    }
  } catch (_) {}
  return null;
}

function getVimeoPlayerConfig(vid, hash) {
  return new Promise((resolve, reject) => {
    const url = `https://player.vimeo.com/video/${vid}${hash ? `?h=${hash}` : ""}`;
    const req = https.get(url, {
      headers: {
        "Referer": "https://app.kinderpedia.co/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
      }
    }, (res) => {
      if (res.statusCode >= 400) {
        return reject(new Error(`Vimeo player returned HTTP ${res.statusCode}`));
      }
      let body = "";
      res.on("data", c => body += c);
      res.on("end", () => {
        const m = body.match(/window\.playerConfig\s*=\s*(\{.*?\});?\s*<\/script>/s);
        if (!m) return reject(new Error("window.playerConfig not found in Vimeo player"));
        try {
          const cfg = JSON.parse(m[1]);
          resolve(cfg);
        } catch (e) {
          reject(e);
        }
      });
    });
    req.on("error", reject);
  });
}

const server = http.createServer(async (req, res) => {
  // Add permissive CORS so app.kinderpedia.co / mykp.kinderpedia.co can call this
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "*");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const parsedUrl = new URL(req.url, `http://localhost:${PORT}`);
  const reqPath = parsedUrl.pathname;

  // --- API Endpoint: /vimeo-config ---
  if (reqPath === "/vimeo-config") {
    if (parsedUrl.searchParams.get("test") === "1") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", service: "local-vimeo-helper" }));
      return;
    }

    let vid = parsedUrl.searchParams.get("id");
    let hash = parsedUrl.searchParams.get("h") || "";
    const rawUrl = parsedUrl.searchParams.get("url");
    if (rawUrl) {
      const mId = rawUrl.match(/video\/(\d+)/) || rawUrl.match(/vimeo\.com\/(\d+)/);
      if (mId) vid = mId[1];
      const mHash = rawUrl.match(/[?&]h=([a-zA-Z0-9]+)/) || rawUrl.match(/vimeo\.com\/\d+\/([a-zA-Z0-9]+)/);
      if (mHash) hash = mHash[1];
    }

    if (!vid) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Missing video id parameter" }));
      return;
    }

    try {
      const cfg = await getVimeoPlayerConfig(vid, hash);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        success: true,
        title: cfg.video?.title || "",
        files: cfg.request?.files || {}
      }));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // --- API Endpoint: /vimeo-stream ---
  if (reqPath === "/vimeo-stream") {
    let vid = parsedUrl.searchParams.get("id");
    let hash = parsedUrl.searchParams.get("h") || "";
    const rawUrl = parsedUrl.searchParams.get("url");
    if (rawUrl) {
      const mId = rawUrl.match(/video\/(\d+)/) || rawUrl.match(/vimeo\.com\/(\d+)/);
      if (mId) vid = mId[1];
      const mHash = rawUrl.match(/[?&]h=([a-zA-Z0-9]+)/) || rawUrl.match(/vimeo\.com\/\d+\/([a-zA-Z0-9]+)/);
      if (mHash) hash = mHash[1];
    }

    if (!vid) {
      res.writeHead(400, { "Content-Type": "text/plain" });
      res.end("Missing video id parameter");
      return;
    }

    try {
      const cfg = await getVimeoPlayerConfig(vid, hash);
      const files = cfg.request?.files || {};

      // 1. Progressive MP4 if available
      const prog = files.progressive;
      if (Array.isArray(prog) && prog.length > 0) {
        prog.sort((a, b) => (b.width || 0) - (a.width || 0));
        const bestMp4 = prog[0].url;
        https.get(bestMp4, (streamRes) => {
          res.writeHead(200, {
            "Content-Type": "video/mp4",
            "Access-Control-Allow-Origin": "*",
          });
          streamRes.pipe(res);
        }).on("error", (e) => {
          if (!res.headersSent) res.writeHead(500);
          res.end(e.message);
        });
        return;
      }

      // 2. HLS stream remuxing via ffmpeg
      const hls = files.hls;
      const defaultCdn = hls?.default_cdn;
      const masterUrl = hls?.cdns?.[defaultCdn]?.avc_url || hls?.cdns?.[defaultCdn]?.url || Object.values(hls?.cdns || {})[0]?.url;

      if (!masterUrl) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("No progressive or HLS stream found for this video");
        return;
      }

      const ffmpeg = findFfmpeg();
      if (!ffmpeg) {
        res.writeHead(501, { "Content-Type": "text/plain" });
        res.end("ffmpeg not found on server host");
        return;
      }

      res.writeHead(200, {
        "Content-Type": "video/mp4",
        "Access-Control-Allow-Origin": "*",
      });

      const proc = spawn(ffmpeg, [
        "-y",
        "-headers", "Referer: https://app.kinderpedia.co/\r\n",
        "-i", masterUrl,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c", "copy",
        "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov",
        "pipe:1"
      ], { stdio: ["ignore", "pipe", "pipe"] });

      proc.stdout.pipe(res);

      req.on("close", () => {
        try { proc.kill("SIGKILL"); } catch (_) {}
      });

      proc.on("error", (err) => {
        console.error("FFmpeg error:", err);
        try { res.end(); } catch (_) {}
      });

    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end(err.message);
      }
    }
    return;
  }

  // --- Static Files ---
  let fileSubpath = reqPath;
  if (fileSubpath === "/") fileSubpath = "/index.html";

  const filePath = path.join(DIR, fileSubpath);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("404 Not Found");
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`🚀 Kinderpedia Bookmarklet server running at http://localhost:${PORT}/`);
  console.log(`👉 Open http://localhost:${PORT}/ to drag the bookmarklet or test the UI.`);
  console.log(`🎬 Video streaming proxy available at /vimeo-stream and /vimeo-config`);
});
