// Service worker: makes the app installable and shows a clear page when the Mac server is not reachable.
// All models run on the Mac, so nothing else is cached: data always comes fresh from the server.
const OFFLINE = "/web/offline.html";
const CACHE = "claim-shell-v5"; // bump whenever offline.html, styles.css or the icon changes
const SHELL = [OFFLINE, "/web/styles.css", "/web/icons/icon-192.png"];

self.addEventListener("install", (e) => {
  // cache: "reload" skips the HTTP cache, so a new version stores the files as they are now
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL.map((u) => new Request(u, { cache: "reload" })))));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", (e) => {
  if (e.request.mode !== "navigate") return; // only page loads get the offline fallback
  e.respondWith(fetch(e.request).catch(() => caches.match(OFFLINE)));
});
