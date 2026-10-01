#!/usr/bin/env python3
"""Assemble le dossier Canard BiscaLab : fragment artifact + PWA complète."""
from pathlib import Path
import re

ROOT = Path(__file__).parent
SRC = ROOT / "src"
PWA = ROOT / "docs"
PWA.mkdir(exist_ok=True)

parts = sorted(SRC.glob("*.html"))
body = "\n".join(p.read_text(encoding="utf-8") for p in parts)
import json
terms = json.loads((SRC / "terms.json").read_text(encoding="utf-8"))
body = body.replace("__TERMS__", json.dumps(terms, ensure_ascii=False, separators=(",", ":")))

# 1) Fragment pour l'Artifact (le viewer ajoute doctype/head/body)
(ROOT / "canard-biscalab.html").write_text(body, encoding="utf-8")

# 2) PWA complète (hébergement statique : Caddy / GitHub Pages)
head_extra = """
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    <meta name="theme-color" content="#e6f1f3" media="(prefers-color-scheme: light)" />
    <meta name="theme-color" content="#1f1d33" media="(prefers-color-scheme: dark)" />
    <meta name="apple-mobile-web-app-capable" content="yes" />
    <meta name="mobile-web-app-capable" content="yes" />
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
    <meta name="apple-mobile-web-app-title" content="Canard" />
    <link rel="manifest" href="manifest.webmanifest" />
    <link rel="icon" type="image/png" sizes="32x32" href="icons/icon-32.png" />
    <link rel="icon" type="image/png" sizes="192x192" href="icons/icon-192.png" />
    <link rel="apple-touch-icon" href="icons/apple-touch-icon.png" sizes="180x180" />
    <script>
      (function () {
        try {
          var t = localStorage.getItem('canard.theme.v1');
          if (t === 'dark' || t === 'light') document.documentElement.setAttribute('data-theme', t);
        } catch (e) {}
      })();
    </script>
"""
# Le fragment commence par <title>, <meta description>, <link fonts>, <style> → tout va dans <head>.
m = re.search(r"</style>\s*", body)
head_part = body[: m.end()]
body_part = body[m.end():]

sw_reg = """
<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('sw.js').catch(function () {});
    });
  }
</script>
"""

index = f"""<!doctype html>
<html lang="fr">
  <head>
    <meta charset="utf-8" />
{head_extra}
{head_part}
  </head>
  <body>
{body_part}
{sw_reg}
  </body>
</html>
"""
(PWA / "index.html").write_text(index, encoding="utf-8")

manifest = """{
  "name": "Canard BiscaLab — Open Duck Mini V2",
  "short_name": "Canard",
  "description": "Dossier de travail du projet Open Duck Mini V2 au BiscaLab",
  "lang": "fr",
  "start_url": "./#accueil",
  "scope": "./",
  "display": "standalone",
  "background_color": "#e6f1f3",
  "theme_color": "#e6f1f3",
  "icons": [
    { "src": "icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any" },
    { "src": "icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any" },
    { "src": "icons/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ]
}
"""
(PWA / "manifest.webmanifest").write_text(manifest, encoding="utf-8")

sw = """/* Canard BiscaLab — service worker : coquille applicative en cache, réseau d'abord pour index.html. */
const VERSION = 'canard-v1.1.0';
const SHELL = ['./', './index.html', './manifest.webmanifest', './icons/icon-192.png', './icons/icon-512.png', './icons/apple-touch-icon.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim())
  );
});
self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  // Polices Google : cache après premier chargement (stale-while-revalidate).
  if (url.origin !== self.location.origin) {
    event.respondWith(
      caches.open(VERSION).then(async (c) => {
        const cached = await c.match(req);
        const fetching = fetch(req).then((res) => { if (res && res.ok) c.put(req, res.clone()); return res; }).catch(() => cached);
        return cached || fetching;
      })
    );
    return;
  }
  // Même origine : réseau d'abord (toujours la dernière version du dossier), repli cache hors ligne.
  event.respondWith(
    fetch(req).then((res) => { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); return res; })
      .catch(() => caches.match(req).then((r) => r || caches.match('./index.html')))
  );
});
"""
(PWA / "sw.js").write_text(sw, encoding="utf-8")

print("fragment:", (ROOT / "canard-biscalab.html").stat().st_size, "octets")
print("index   :", (PWA / "index.html").stat().st_size, "octets")
