// SwingLab offline cache. Bump VERSION when you publish changes.
const VERSION = 'swinglab-v2';
const CORE = ['./', './index.html', './manifest.webmanifest', './models/model.json', './models/movenet-lightning-weights.wasm', './icons/icon-192.png', './icons/icon-512.png', './icons/apple-touch-icon.png'];
const EXTRA = ['https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(VERSION).then(async c => { await c.addAll(CORE); for (const u of EXTRA) { try { await c.add(new Request(u, { mode: 'cors' })); } catch {} } }).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== VERSION).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request; if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (req.mode === 'navigate') {
    e.respondWith(fetch(req).then(r => { const c = r.clone(); caches.open(VERSION).then(x => x.put('./index.html', c)); return r; }).catch(() => caches.match('./index.html')));
    return;
  }
  const cacheable = url.origin === location.origin || /(^|\.)jsdelivr\.net$|fonts\.(googleapis|gstatic)\.com$/.test(url.hostname);
  if (!cacheable) return;
  e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(r => { if (r.ok) { const c = r.clone(); caches.open(VERSION).then(x => x.put(req, c)); } return r; })));
});
