/**
 * Cloudflare Worker: Vimeo Video Helper for Kinderpedia Backup
 * 
 * Purpose:
 *   Relays the initial 5 KB Vimeo embed configuration so the browser bookmarklet
 *   can download videos directly without CORS blocking.
 * 
 * Security through Obscurity:
 *   Requires a shared secret (via query string `?secret=...` or header `X-KP-Secret`).
 *   Unauthorized requests receive a silent `404 Not Found` to prevent discovery.
 */

// Default shared secret (can also be set in Cloudflare Worker Settings -> Variables -> SECRET)
const DEFAULT_SECRET = "kp_sec_98721";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 1. Handle CORS preflight for browser fetch calls
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, X-KP-Secret",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    // 2. Security through obscurity: Shared secret verification
    const expectedSecret = (env && env.SECRET) || DEFAULT_SECRET;
    const providedSecret = url.searchParams.get("secret") || request.headers.get("x-kp-secret");

    if (!providedSecret || providedSecret !== expectedSecret) {
      // Return 404 Not Found so port scanners and bots think this endpoint does not exist
      return new Response("404 Not Found", {
        status: 404,
        headers: { "Content-Type": "text/plain" },
      });
    }

    // 3. Health check / ping
    if (url.searchParams.get("ping") === "1" || url.pathname === "/ping") {
      return new Response(JSON.stringify({ status: "ok", service: "kinderpedia-vimeo-worker" }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }

    // 4. Extract video ID and unlisted hash
    let vid = url.searchParams.get("id");
    let hash = url.searchParams.get("h") || "";
    const rawUrl = url.searchParams.get("url");

    if (rawUrl) {
      const mId = rawUrl.match(/video\/(\d+)/) || rawUrl.match(/vimeo\.com\/(\d+)/);
      if (mId) vid = mId[1];
      const mHash = rawUrl.match(/[?&]h=([a-zA-Z0-9]+)/) || rawUrl.match(/vimeo\.com\/\d+\/([a-zA-Z0-9]+)/);
      if (mHash) hash = mHash[1];
    }

    if (!vid) {
      return new Response(JSON.stringify({ error: "Missing video id parameter (?id=12345678)" }), {
        status: 400,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }

    // 5. Fetch Vimeo embed player page to extract configuration
    try {
      const embedUrl = `https://player.vimeo.com/video/${vid}${hash ? `?h=${hash}` : ""}`;
      const vimeoRes = await fetch(embedUrl, {
        headers: {
          "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          "Referer": "https://app.kinderpedia.co/",
          "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
      });

      if (!vimeoRes.ok) {
        return new Response(JSON.stringify({ error: `Vimeo returned HTTP ${vimeoRes.status}` }), {
          status: vimeoRes.status,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        });
      }

      const html = await vimeoRes.text();
      const m = html.match(/window\.playerConfig\s*=\s*(\{.+?\})(?:;|\s*<\/script>)/s) ||
                html.match(/(?:window\.playerConfig|var config)\s*=\s*(\{.+?\});/s);

      if (!m) {
        return new Response(JSON.stringify({ error: "Could not locate playerConfig in Vimeo page" }), {
          status: 502,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        });
      }

      const cfg = JSON.parse(m[1]);
      const files = cfg.request?.files || {};

      // Return clean, CORS-enabled JSON payload to the bookmarklet
      return new Response(JSON.stringify({
        success: true,
        id: vid,
        title: cfg.video?.title || "",
        duration: cfg.video?.duration || 0,
        progressive: files.progressive || null,
        hls: files.hls || null,
        raw_files: files,
      }), {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
          "Cache-Control": "public, max-age=1800",
        },
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: err.message || "Failed to fetch Vimeo configuration" }), {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    }
  },
};
