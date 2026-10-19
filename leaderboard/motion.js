'use strict';
/*
 * Page motion shared with the main project site (ported from web_v2/src/motion/cursor.ts,
 * scroll.ts and sectionRail.ts, same constants and class names):
 *  - cursor follower: a thin ring trailing the pointer and a small dot (mouse/pen only);
 *  - header: tucks away while scrolling down, comes back on scrolling up or when the pointer
 *    reaches the top band of the window;
 *  - section rail (bottom-left): one diamond per section, the current one a bar filling with
 *    progress and its name beside it. Sections are the elements marked data-rail="Label".
 * Everything is off with reduced motion where the main site turns it off.
 */
(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- Cursor follower ----------
  function initCursor() {
    if (!matchMedia('(pointer: fine)').matches || reduced) return;
    const RING_FOLLOW = 9, DOT_FOLLOW = 30;
    const INTERACTIVE = 'a, button, summary, [role="button"], input, select, textarea, label, [data-tier-model]';
    const ring = document.createElement('div'), dot = document.createElement('div');
    ring.className = 'cursor-ring'; dot.className = 'cursor-dot';
    ring.setAttribute('aria-hidden', 'true'); dot.setAttribute('aria-hidden', 'true');
    document.body.append(ring, dot);
    const target = { x: innerWidth / 2, y: innerHeight / 2 }, r = { ...target }, d = { ...target };
    let shown = false, raf = 0, last = 0;
    const place = () => {
      ring.style.transform = `translate3d(${r.x}px, ${r.y}px, 0) translate(-50%, -50%)`;
      dot.style.transform = `translate3d(${d.x}px, ${d.y}px, 0) translate(-50%, -50%)`;
    };
    const tick = (now) => {
      const dt = Math.min((now - last) / 1000, 0.05); last = now;
      const kr = 1 - Math.exp(-dt * RING_FOLLOW), kd = 1 - Math.exp(-dt * DOT_FOLLOW);
      r.x += (target.x - r.x) * kr; r.y += (target.y - r.y) * kr;
      d.x += (target.x - d.x) * kd; d.y += (target.y - d.y) * kd;
      place();
      raf = Math.abs(target.x - r.x) + Math.abs(target.y - r.y) > 0.1 ? requestAnimationFrame(tick) : 0;
    };
    const wake = () => { if (raf) return; last = performance.now(); raf = requestAnimationFrame(tick); };
    addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse' && e.pointerType !== 'pen') return;
      target.x = e.clientX; target.y = e.clientY;
      if (!shown) { r.x = d.x = target.x; r.y = d.y = target.y; place(); shown = true; document.documentElement.classList.add('has-cursor'); }
      wake();
    }, { passive: true });
    document.documentElement.addEventListener('pointerleave', () => { shown = false; document.documentElement.classList.remove('has-cursor'); });
    addEventListener('pointerover', (e) => ring.classList.toggle('is-hover', !!e.target?.closest?.(INTERACTIVE)), { passive: true });
    addEventListener('pointerdown', () => ring.classList.add('is-down'), { passive: true });
    addEventListener('pointerup', () => ring.classList.remove('is-down'), { passive: true });
  }

  // ---------- Header: hide on scroll down, show on scroll up / pointer at the top ----------
  function initHeader() {
    const header = document.querySelector('.site-header'); if (!header) return;
    const SCROLL_STEP = 6, REVEAL_BAND = 72;
    const navHeight = () => parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nav-height')) || 96;
    let lastY = scrollY, scrolledAway = false, pointerReveal = false, pending = false;
    const navTargets = [...header.querySelectorAll('nav > a[href^="#"]')].map((a) => ({ a, el: document.querySelector(a.hash) })).filter((t) => t.el);
    const apply = () => header.classList.toggle('is-hidden', scrolledAway && !pointerReveal);
    const update = () => {
      pending = false;
      const dy = scrollY - lastY;
      if (Math.abs(dy) >= SCROLL_STEP) { scrolledAway = dy > 0 && scrollY > navHeight(); lastY = scrollY; }
      if (scrollY <= 0) scrolledAway = false;
      apply();
      // Nav links to sections on this page get the "current" underline.
      const offset = navHeight() + 100;
      const current = navTargets.filter((t) => t.el.getBoundingClientRect().top <= offset).at(-1);
      navTargets.forEach((t) => { if (t === current) t.a.setAttribute('aria-current', 'location'); else t.a.removeAttribute('aria-current'); });
    };
    addEventListener('scroll', () => { if (!pending) { pending = true; requestAnimationFrame(update); } }, { passive: true });
    addEventListener('pointermove', (e) => {
      if (e.pointerType === 'touch') return;
      const band = Math.max(REVEAL_BAND, pointerReveal ? header.getBoundingClientRect().bottom : 0);
      const reveal = e.clientY <= band;
      if (reveal !== pointerReveal) { pointerReveal = reveal; apply(); }
    }, { passive: true });
    document.documentElement.addEventListener('pointerleave', () => { if (pointerReveal) { pointerReveal = false; apply(); } });
    update();
  }

  // ---------- Section rail (bottom-left) ----------
  function initSectionRail() {
    const READING_LINE = 0.4;
    const all = [...document.querySelectorAll('[data-rail]')].map((el) => ({ el, label: el.dataset.rail, id: el.id }));
    if (all.length < 2) return;
    const nav = document.createElement('nav');
    nav.className = 'section-rail'; nav.setAttribute('aria-label', 'Section progress');
    nav.innerHTML = `<ol>${all.map((s, i) => `<li><a href="#${s.id}" data-rail-index="${i}"><span class="section-rail-node" aria-hidden="true"><i></i></span><span class="section-rail-label">${s.label}</span></a></li>`).join('')}</ol>`;
    document.body.append(nav);
    const anchors = [...nav.querySelectorAll('a')];
    let tops = [], live = [], pageEnd = 0, pending = false;
    const measure = () => {
      // sections that are hidden (e.g. not filled yet) leave the rail
      live = all.map((s) => !s.el.hidden && s.el.offsetParent !== null);
      anchors.forEach((a, i) => { a.parentElement.hidden = !live[i]; });
      tops = all.map((s) => s.el.getBoundingClientRect().top + scrollY);
      pageEnd = document.documentElement.scrollHeight - innerHeight * (1 - READING_LINE);
    };
    const update = () => {
      pending = false;
      const y = scrollY + innerHeight * READING_LINE, idx = all.map((_, i) => i).filter((i) => live[i]);
      let current = -1;
      idx.forEach((i) => { if (y >= tops[i]) current = i; });
      if (current >= 0) {
        const k = idx.indexOf(current), next = idx[k + 1];
        const from = tops[current], to = next != null ? tops[next] : pageEnd;
        anchors[current].style.setProperty('--p', Math.min(1, Math.max(0, (y - from) / Math.max(1, to - from))).toFixed(4));
      }
      anchors.forEach((a, i) => {
        a.classList.toggle('is-current', i === current);
        a.classList.toggle('is-passed', current >= 0 && i < current);
        if (i === current) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
      });
      nav.classList.toggle('is-shown', current >= 0);
    };
    const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(update); } };
    measure(); update();
    addEventListener('scroll', schedule, { passive: true });
    addEventListener('resize', () => { measure(); schedule(); });
    if (typeof ResizeObserver === 'function') new ResizeObserver(() => { measure(); schedule(); }).observe(document.body);
  }

  // ---------- Scroll flow (main site: motion/cinematic/index.ts FLOW + immersive.css) ----------
  // Each block gets two scrubbed variables: --in 0→1 while its top travels from 96% to 58% of
  // the viewport (ease power2.out), --out 0→1 while its bottom travels from 40% to 2% (power2.in).
  // CSS turns them into opacity, rise and scale; section headings also un-blur. The values trail
  // the scroll (≈0.6 s), like GSAP's scrub: 0.6.
  function initFlow() {
    if (reduced) return;
    const SELECTOR = [
      'main > section:not(.hero):not(.model-hero) > :not(.reports-bento):not(.editorial-grid-head):not(.findings-editorial)',
      // model report findings: each ticket enters and leaves on its own
      '.findings-editorial .fe-row',
      // model report pages: an editorial grid's heading column uses display: contents, so its
      // index, title and intro are the blocks
      '.update-page .editorial-grid > div:first-child > *',
      '.reports-bento > *',
      '.news-page > .breadcrumbs',
      '.news-page > section > *',
    ].join(',');
    let blocks = [];
    const collect = () => {
      blocks = [...document.querySelectorAll(SELECTOR)].filter((el) => !el.closest('dialog') && el.offsetParent !== null);
      blocks.forEach((el) => { if (!el.hasAttribute('data-flow')) { el.setAttribute('data-flow', ''); el._in = el._out = null; } });
    };
    // Windows are clamped to where a block sits at the top of the page, so content on the first
    // screen starts fully in and only leaves once it is scrolled past.
    const target = (el) => {
      // Leaving is gentler than the main site's (on request, 2026-10-17): a block starts to fade
      // once its bottom passes 22% of the viewport and is gone 34% of a viewport later.
      const r = el.getBoundingClientRect(), vh = innerHeight, span = vh * 0.38, outSpan = vh * 0.34;
      const docTop = r.top + scrollY, docBottom = r.bottom + scrollY;
      const inEnd = Math.min(vh * 0.58, docTop), outStart = Math.min(vh * 0.22, docBottom);
      const ti = Math.min(1, Math.max(0, (inEnd + span - r.top) / span));
      const to = Math.min(1, Math.max(0, (outStart - r.bottom) / outSpan));
      return [1 - (1 - ti) * (1 - ti), to * to];
    };
    let raf = 0, last = 0;
    const tick = (now) => {
      const k = 1 - Math.exp(-Math.min(0.05, (now - last) / 1000) / 0.2); last = now;
      let moving = false;
      for (const el of blocks) {
        const [a, b] = target(el);
        if (el._in == null) { el._in = a; el._out = b; }
        el._in += (a - el._in) * k; el._out += (b - el._out) * k;
        if (Math.abs(a - el._in) + Math.abs(b - el._out) > 0.002) moving = true; else { el._in = a; el._out = b; }
        el.style.setProperty('--in', el._in.toFixed(4)); el.style.setProperty('--out', el._out.toFixed(4));
      }
      raf = moving ? requestAnimationFrame(tick) : 0;
    };
    const wake = () => { if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); } };
    collect();
    document.documentElement.classList.add('has-flow');
    tick(performance.now());
    addEventListener('scroll', wake, { passive: true });
    addEventListener('resize', wake);
    // content filled in later (rankings, reports, cases) joins the flow
    if (typeof MutationObserver === 'function') new MutationObserver(() => { collect(); wake(); }).observe(document.querySelector('main') || document.body, { childList: true, subtree: true });
  }

  // A refresh starts at the top of the page: section links scroll without writing their #anchor
  // into the address bar, and a reload drops a section anchor left there. Deep links that carry
  // state (#model=…, #case-…) and links opened from elsewhere still go to their target.
  function initAnchors() {
    const g = globalThis;
    if (!g.history || !g.location || !g.addEventListener) return;
    const sectionHash = (h) => /^#[A-Za-z][\w-]*$/.test(h || '') && !/^#case-\d+$/.test(h);
    const nav = g.performance?.getEntriesByType?.('navigation')?.[0];
    if ('scrollRestoration' in g.history) g.history.scrollRestoration = 'manual';
    if (nav?.type === 'reload') {
      if (sectionHash(g.location.hash)) g.history.replaceState(null, '', g.location.pathname + g.location.search);
      if (!g.location.hash) g.scrollTo?.(0, 0);
    }
    document.addEventListener('click', (e) => {
      const a = e.target?.closest?.('a[href^="#"]');
      if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey) return;
      const h = a.getAttribute('href'); if (!sectionHash(h)) return;
      const sec = document.getElementById(h.slice(1)); if (!sec) return;
      // land on the section's opening rule (its heading), not on the padding above it
      const el = sec.querySelector('.section-heading, .bento-top, .cs-head') || sec;
      e.preventDefault();
      // layout position (offsetTop ignores the flow transforms) minus the html scroll-padding
      const top = () => { let y = 0; for (let n = el; n; n = n.offsetParent) y += n.offsetTop; return y - (parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0); };
      g.scrollTo({ top: top(), behavior: g.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
      // late layout shifts (lazy images, filled-in data) can stop a smooth scroll short: settle once more
      g.addEventListener('scrollend', () => { if (Math.abs(g.scrollY - top()) > 2) g.scrollTo({ top: top() }); }, { once: true });
    });
  }

  // cursor follower switched off on the leaderboard pages (2026-10-18); initCursor stays for reuse
  const start = () => { initHeader(); initSectionRail(); initFlow(); initAnchors(); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
