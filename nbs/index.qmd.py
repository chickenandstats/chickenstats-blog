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


def blog_card(im):
    """Quarto blog card."""
    return qmd.div(img(f"cards/{im}.png"), ["carousel-list"])


blog_cards = ["predators_card", "titans_card", "vols_card", "other_card"]

blog_cards_d = qmd.div("\n".join([blog_card(x) for x in blog_cards]), ["carousel-box"])


###
# Output section
###

d(
    """# chickenstats-blog

### The python code and notebooks behind the newsletter and library""",
    "hero-banner",
)

d(img("site/hero_transparent_alt.png"), "hero-banner")

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

d(blog_cards_d, "carousel-box")
