document.addEventListener('DOMContentLoaded', function () {
    /* ---------- mark JS active so hidden-state CSS only applies when JS runs ---------- */
    document.documentElement.classList.add('js-enabled');

    /* ---------- prefs (cookies; work in private mode too) ---------- */
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function getCookie(n) {
        const m = document.cookie.match(new RegExp('(?:^|; )' + n + '=([^;]*)'));
        return m ? decodeURIComponent(m[1]) : null;
    }
    function setCookie(n, v) {
        document.cookie = n + '=' + encodeURIComponent(v) + ';path=/;max-age=31536000;SameSite=Lax';
    }

    /* ---------- theme: apply saved choice on load ---------- */
    const savedTheme = getCookie('theme');
    if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);

    /* ---------- hamburger overlay menu ---------- */
    const hamburger = document.getElementById('hamburger');
    const menu = document.getElementById('menu');
    const links = document.querySelectorAll('.menu-link');
    let open = false;

    if (hamburger && menu) {
        hamburger.addEventListener('click', function () {
            open = !open;
            menu.classList.toggle('open', open);
            // Class on the hamburger ITSELF so the bar -> X animation
            // works regardless of DOM sibling order.
            hamburger.classList.toggle('is-open', open);
            hamburger.setAttribute('aria-expanded', String(open));

            if (open) {
                links.forEach((l, i) => setTimeout(() => l.classList.add('show'), 150 + i * 80));
            } else {
                links.forEach(l => l.classList.remove('show'));
            }
        });
    }

    /* ---------- custom cursor (fine pointer only) ---------- */
    const dot = document.getElementById('cursor-dot');
    const finePointer = window.matchMedia('(pointer: fine)').matches;

    if (dot && finePointer) {
        document.addEventListener('mousemove', (e) => {
            dot.style.left = e.clientX + 'px';
            dot.style.top = e.clientY + 'px';
        });
        document.querySelectorAll('a, button').forEach(el => {
            el.addEventListener('mouseenter', () => dot.classList.add('hover'));
            el.addEventListener('mouseleave', () => dot.classList.remove('hover'));
        });
    } else if (dot) {
        dot.style.display = 'none';
        document.documentElement.classList.add('no-custom-cursor');
    }

    /* ---------- scroll reveal ---------- */
    const revealEls = document.querySelectorAll('.reveal');
    const animOff = getCookie('anim') === 'off' || prefersReduced;

    function revealAll() {
        revealEls.forEach(el => el.classList.add('show'));
    }

    if (animOff || !('IntersectionObserver' in window)) {
        // No animation wanted, or observer unsupported: show everything immediately.
        revealAll();
    } else {
        const revealObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('show');
                    revealObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0, rootMargin: '0px 0px -10% 0px' });

        revealEls.forEach(el => revealObserver.observe(el));

        // Safety net: if anything is still hidden shortly after load
        // (e.g. tall block that never crossed threshold), force it visible.
        window.addEventListener('load', () => {
            setTimeout(() => {
                revealEls.forEach(el => {
                    const r = el.getBoundingClientRect();
                    if (r.top < window.innerHeight && !el.classList.contains('show')) {
                        el.classList.add('show');
                    }
                });
            }, 400);
        });
    }

    /* ---------- copy buttons on code blocks ---------- */
    document.querySelectorAll('.code-block').forEach(block => {
        const code = block.querySelector('code');
        if (!code) return;
        const btn = document.createElement('button');
        btn.textContent = 'Copy';
        btn.className = 'copy-btn';
        btn.addEventListener('click', () => {
            navigator.clipboard.writeText(code.textContent).then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => (btn.textContent = 'Copy'), 1200);
            });
        });
        block.appendChild(btn);
    });

    /* ---------- theme toggle ---------- */
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        const setIcon = () => {
            const light = document.documentElement.getAttribute('data-theme') === 'light';
            themeToggle.textContent = light ? '\u2600' : '\u263E';
            themeToggle.setAttribute('aria-label', light ? 'Switch to dark mode' : 'Switch to light mode');
        };
        setIcon();
        themeToggle.addEventListener('click', () => {
            const light = document.documentElement.getAttribute('data-theme') === 'light';
            const next = light ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', next);
            setCookie('theme', next);
            setIcon();
        });
    }

    /* ---------- typewriter ---------- */
    function typeWriter(el) {
        const text = el.getAttribute('data-typewriter') || el.textContent;
        const speed = parseInt(el.getAttribute('data-speed'), 10) || 60;
        el.textContent = '';
        el.classList.add('typing');
        let i = 0;
        (function step() {
            if (i < text.length) {
                el.textContent += text.charAt(i++);
                setTimeout(step, speed);
            } else {
                el.classList.remove('typing');
                el.classList.add('typed');
            }
        })();
    }

    const typeTargets = document.querySelectorAll('[data-typewriter]');
    typeTargets.forEach(el => {
        if (animOff) {
            el.textContent = el.getAttribute('data-typewriter') || el.textContent;
            el.classList.add('typed');
        } else {
            typeWriter(el);
        }
    });

    /* ---------- animation ("too laggy") toggle ---------- */
    const animToggle = document.getElementById('anim-toggle');
    if (animToggle) {
        const reflect = () => {
            const off = getCookie('anim') === 'off';
            animToggle.textContent = off ? 'Animations: off' : 'Animations: on';
            animToggle.setAttribute('aria-pressed', String(off));
        };
        reflect();
        animToggle.addEventListener('click', () => {
            const off = getCookie('anim') === 'off';
            setCookie('anim', off ? 'on' : 'off');
            reflect();
            // Ensure content is visible regardless of new state.
            revealAll();
            typeTargets.forEach(el => {
                el.textContent = el.getAttribute('data-typewriter') || el.textContent;
                el.classList.remove('typing');
                el.classList.add('typed');
            });
        });
    }
});