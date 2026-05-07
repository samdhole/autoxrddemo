---
tokens:
  color:
    bg: "#f7f5ef"
    paper: "#fffefa"
    paper-strong: "#f0eee6"
    ink: "#191b1d"
    muted: "#5f676b"
    supporting-text: "#30363a"
    line: "#d8d4c8"
    accent: "#12605a"
    accent-strong: "#0d4743"
    accent-soft: "#dcebe8"
    blue: "#254b6b"
    warn: "#7a5a15"
  radius:
    default: "8px"
    pill: "999px"
  shadow:
    card: "0 18px 45px rgba(26, 31, 35, 0.08)"
  layout:
    max-width: "1160px"
    section-padding: "76px 22px"
    nav-height: "64px"
    scroll-margin-top: "88px"
  typography:
    body-family: "Arial, Helvetica, sans-serif"
    heading-family: "Georgia, 'Times New Roman', serif"
    line-height-body: "1.55"
    line-height-heading: "1.08"
    size-h1: "clamp(3rem, 8vw, 5.9rem)"
    size-hero-lead: "clamp(1.65rem, 3.5vw, 2.6rem)"
    size-supporting: "clamp(1.05rem, 1.8vw, 1.28rem)"
    size-nav: "0.9rem"
    size-eyebrow: "0.76rem"
    size-body: "1rem"
    weight-heading: "700"
    weight-eyebrow: "800"
    weight-brand: "800"
---

# DESIGN.md — Sam Dhole Personal Site

## Identity

**Name displayed:** Sam Dhole, PhD
**Tagline:** Materials Scientist for Technical Documents
**Tone:** Credible professional document, not a startup landing page. Warm, dense, direct.

The site should feel like a well-typeset academic CV crossed with a consultant's calling card. No hero illustrations, no stock photos, no gradient blobs.

## Color

| Role | Token | Hex |
|---|---|---|
| Page background | `--bg` | `#f7f5ef` warm parchment |
| Card / panel surface | `--paper` | `#fffefa` near-white |
| Card strong background | `--paper-strong` | `#f0eee6` |
| Primary text | `--ink` | `#191b1d` near-black |
| Secondary text | `--muted` | `#5f676b` |
| Supporting body copy | _(inline)_ | `#30363a` |
| Borders and dividers | `--line` | `#d8d4c8` |
| Primary accent (teal) | `--accent` | `#12605a` |
| Accent dark (buttons, headings) | `--accent-strong` | `#0d4743` |
| Accent light (pill backgrounds) | `--accent-soft` | `#dcebe8` |
| Blue (links, secondary) | `--blue` | `#254b6b` |
| Amber warning | `--warn` | `#7a5a15` |

The accent teal is the primary brand color. Use it for CTAs, active states, eyebrow labels, and hover accents. Do not introduce other hues.

## Typography

**Body:** Arial, Helvetica, sans-serif. 1rem / 1.55 line-height. Color: `--ink`.

**Headings (h1, h2, h3):** Georgia, serif. Line-height 1.08. Letter-spacing 0. Color: `--ink`.

| Element | Size | Weight | Family |
|---|---|---|---|
| h1 | clamp(3rem, 8vw, 5.9rem) | 700 | Georgia serif |
| hero-lead | clamp(1.65rem, 3.5vw, 2.6rem) | normal | Georgia serif |
| Supporting / body large | clamp(1.05rem, 1.8vw, 1.28rem) | normal | Arial |
| Nav links | 0.9rem | normal | Arial |
| Eyebrow / tag / status pill | 0.76rem | 800 | Arial |
| Brand / logo | inherit | 800 | Arial |

No web fonts. System fonts only: Arial for body, Georgia for display.

## Spacing and Layout

- Max content width: 1160px, centered.
- Section vertical padding: 76px 22px.
- Nav height: 64px sticky. Scroll margin on sections: 88px.
- Card border-radius: 8px. Pill radius: 999px.
- Card shadow: `0 18px 45px rgba(26, 31, 35, 0.08)`.
- Hero grid: 2 columns — `minmax(0, 1.25fr)` left, `minmax(280px, 0.75fr)` right. Full viewport height minus nav.
- Mobile: single column, stack all grids.

## Components

### Nav
Sticky, blurred parchment (`rgba(247, 245, 239, 0.96)` + `backdrop-filter: blur(12px)`). Bottom border `--line`. Brand left, links right. CV appears as a small pill-outlined button in accent-strong. Icon links use 28x28 rounded squares.

### Eyebrow / Tag / Status Pills
Pill shape (`border-radius: 999px`). Background `--accent-soft`. Text `--accent-strong`. Border `rgba(18, 96, 90, 0.24)`. Font 0.76rem / weight 800 / uppercase. Used for availability signal, section labels, and tech tags on cards.

### Buttons
Min-height 46px. Inline-flex. Two variants:
- **Primary:** Background `--accent`, color white, border-radius 8px. Hover: `--accent-strong`.
- **Outline / secondary:** Border `--line`, color `--ink`, border-radius 8px. Hover: light background.

CV download: pill-outlined in `--accent-strong`, weight 700.

### Cards (case studies, services, proof)
Background `--paper`. Border 1px `--line`. Border-radius 8px. Shadow `--shadow`. Padding approx 24-28px. Tags below the card title use eyebrow pill style. Cards are clickable only when a real page or artifact exists — do not fake card links.

### Focus states
Outline: `3px solid rgba(18, 96, 90, 0.36)`, offset 3px. Applies to all interactive elements.

### Hero
Two-column grid on desktop. Left: identity + headline + supporting line + CTA row. Right: compact credential / proof panel. Background: `linear-gradient(180deg, #fffdf6 0%, var(--bg) 100%)`. Bottom border `--line`. No decorative illustration.

### Section headers
h2 Georgia serif. Optional eyebrow pill above. Left-aligned, not centered, unless a centering reason exists.

## Copy Rules

- No em-dashes. Use commas or colons instead.
- No generic "AI consultant" language.
- No invented metrics, testimonials, or case-study links.
- Concrete first projects, not vague consulting.
- Public-safe proof only.
- Positioning: "Materials Scientist for Technical Documents."

## What Not to Do

- Do not add gradient blobs, hero illustrations, or stock photography.
- Do not introduce new brand colors beyond the token set above.
- Do not use web fonts or Google Fonts.
- Do not center-align body text or section headers by default.
- Do not use animated nav or heavy scroll effects.
- Do not create fake links on case-study cards before real artifacts exist.
- Do not use startup/SaaS tone, agency/team language, or aggressive CTA copy.
