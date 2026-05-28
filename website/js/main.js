// DECOMP report — reveal-on-scroll, floating TOC active state, back-to-top.

(function () {
  'use strict';

  /* ---- Reveal on scroll ---- */
  const revealEls = document.querySelectorAll('.reveal');
  const revealObs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        revealObs.unobserve(e.target);
      }
    }
  }, { rootMargin: '0px 0px -10% 0px', threshold: 0.05 });
  revealEls.forEach((el) => revealObs.observe(el));

  /* ---- Floating TOC (visible after scrolling past hero, hidden in footer) ---- */
  const toc = document.querySelector('.floating-toc');
  const hero = document.querySelector('.hero');
  const footer = document.querySelector('footer');
  function updateTocVisibility() {
    if (!toc || !hero) return;
    const past = window.scrollY > hero.offsetHeight * 0.6;
    const nearFooter = footer && (window.innerHeight + window.scrollY > footer.offsetTop - 80);
    toc.classList.toggle('visible', past && !nearFooter);
  }

  /* ---- TOC active-link highlight ---- */
  const tocLinks = toc ? Array.from(toc.querySelectorAll('a')) : [];
  const tocTargets = tocLinks.map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  function updateActiveToc() {
    if (!tocTargets.length) return;
    const scroll = window.scrollY + 120;
    let activeIdx = 0;
    for (let i = 0; i < tocTargets.length; i++) {
      if (tocTargets[i].offsetTop <= scroll) activeIdx = i;
    }
    tocLinks.forEach((l, i) => l.classList.toggle('active', i === activeIdx));
  }

  /* ---- Back-to-top ---- */
  const btt = document.querySelector('.back-to-top');
  function updateBtt() {
    if (!btt) return;
    btt.classList.toggle('visible', window.scrollY > 600);
  }

  function onScroll() {
    updateTocVisibility();
    updateActiveToc();
    updateBtt();
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  onScroll();
})();
