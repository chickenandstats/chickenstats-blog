"""---
pagetitle: chickenstats-blog
page-layout: custom
section-divs: false
css: index.css
toc: false
image: https://raw.githubusercontent.com/chickenandstats/chickenstats-blog/refs/heads/main/nbs/images/site/hero_transparent.png.

---
"""

from fastcore.foundation import L
from nbdev import qmd


def img(fname, classes=None, **kwargs):
    """Quarto image."""
    return qmd.img(f"images/{fname}", classes=classes, **kwargs)


def btn(txt, link):
    """Quarto button."""
    return qmd.btn(txt, link=link, classes=["btn-action-primary", "btn-action", "btn", "btn-success", "btn-lg"])


def banner(txt, classes=None, style=None):
    """Quarto banner."""
    return qmd.div(txt, L("hero-banner") + classes, style=style)


def b(*args, **kwargs):
    """Quarto banner."""
    print(banner(*args, **kwargs))


def d(*args, **kwargs):
    """Quarto div."""
    print(qmd.div(*args, **kwargs))


###
# Output section
###

d(
    """# chickenstats-blog

### The python code and notebooks behind the newsletter and library""",
    "hero-banner",
)

d(img("site/hero_transparent_thin.png"), "hero-banner")

d(
    """

<div style="min-height: 58px;max-width: 440px;margin: 0 auto;width: 100%">
<script src="https://cdn.jsdelivr.net/ghost/signup-form@~0.2/umd/signup-form.min.js" 
data-button-color="#4051b5" data-button-text-color="#FFFFF" data-site="https://chickenandstats.com/" 
data-locale="en" async></script>
</div>

""",
    "hero-banner",
)
