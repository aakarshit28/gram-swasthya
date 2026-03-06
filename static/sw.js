/* ================================================================
   GramSwasthya Mitra — Service Worker
   Cache-first for static assets, network-first for API/pages
   ================================================================ */

const CACHE_NAME = 'gsm-v1';
const STATIC_ASSETS = [
  '/',
  '/home',
  '/static/style.css',
  '/static/manifest.json',
  '/login_page',
  '/register_page'
];

// ── INSTALL: Pre-cache static assets ─────────────────────────
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(function(cache) {
      return cache.addAll(STATIC_ASSETS).catch(function(err) {
        console.log('[SW] Pre-cache failed for some assets:', err);
      });
    }).then(function() {
      return self.skipWaiting();
    })
  );
});

// ── ACTIVATE: Clean up old caches ────────────────────────────
self.addEventListener('activate', function(event) {
  event.waitUntil(
    caches.keys().then(function(keys) {
      return Promise.all(
        keys.filter(function(key) { return key !== CACHE_NAME; })
            .map(function(key) { return caches.delete(key); })
      );
    }).then(function() {
      return self.clients.claim();
    })
  );
});

// ── FETCH: Cache-first for static, network-first for API ─────
self.addEventListener('fetch', function(event) {
  var url = new URL(event.request.url);

  // API calls — always try network first, no caching
  if (url.pathname.startsWith('/predict') ||
      url.pathname.startsWith('/translate') ||
      url.pathname.startsWith('/login') ||
      url.pathname.startsWith('/register') ||
      url.pathname.startsWith('/logout') ||
      url.pathname.startsWith('/history') ||
      url.pathname.startsWith('/save_assessment') ||
      url.pathname.startsWith('/disease_info') ||
      url.pathname.startsWith('/get_user') ||
      url.pathname.startsWith('/asha')) {
    event.respondWith(
      fetch(event.request).catch(function() {
        return new Response(JSON.stringify({ error: 'offline', message: 'No internet connection' }), {
          headers: { 'Content-Type': 'application/json' }
        });
      })
    );
    return;
  }

  // Static assets — cache-first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then(function(cached) {
        return cached || fetch(event.request).then(function(response) {
          return caches.open(CACHE_NAME).then(function(cache) {
            cache.put(event.request, response.clone());
            return response;
          });
        });
      })
    );
    return;
  }

  // HTML pages — network-first with offline fallback
  event.respondWith(
    fetch(event.request).then(function(response) {
      // Cache successful page loads
      return caches.open(CACHE_NAME).then(function(cache) {
        cache.put(event.request, response.clone());
        return response;
      });
    }).catch(function() {
      // Offline fallback — serve cached version
      return caches.match(event.request).then(function(cached) {
        if (cached) return cached;
        // Fallback to cached home page
        return caches.match('/home').then(function(home) {
          return home || caches.match('/');
        });
      });
    })
  );
});

// ── Handle background sync (future enhancement) ──────────────
self.addEventListener('message', function(event) {
  if (event.data === 'skipWaiting') {
    self.skipWaiting();
  }
});
