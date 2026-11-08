# Design system: how FilingSight applies ai-design-skills

The UI follows [elayadesign/ai-design-skills](https://github.com/elayadesign/ai-design-skills) (MIT, commit `456d9e0`).
Both skills are vendored in [`.claude/skills/`](../.claude/skills/) so any AI coding agent working in this repo loads the same rules.

| Skill | Rule | Where it shows up in `app.py` |
|---|---|---|
| landing-page-design | **A2** Above the fold: headline, subheadline, one primary action, one proof signal, hero visual | Hero headline and subheadline, the chat box as the one action, KPI row as proof (hit@5 0.95), and a real cited answer card as the hero visual |
| landing-page-design | **A5 / B8** Specific copy, no filler words, never invent facts | Every number on screen comes from `eval/results/*.json` or the filings (for example Uber revenue +57%, page 57). No "revolutionize" or "supercharge" |
| landing-page-design | **B1** One typeface, no italics, no weights above bold, Geist Mono only for numbers | Manrope everywhere, Geist Mono for KPIs, scores and code; `em, i {font-style: normal}`; weights 400 to 600 |
| landing-page-design | **B1** Type scale and named text styles | Sizes are only 12/14/16/18/20/24/30/36/48 px with their paired line heights (Caption, Body, Title, Heading steps) |
| landing-page-design | **B1** No orphaned words | `text-wrap: balance` on headings, `pretty` on body; hero and tagline lines broken by hand at meaningful points |
| landing-page-design | **B2** Spacing tokens | Padding, gaps and margins use only 2, 4, 8, 12, 16, 24, 32, 48, 64, 96 px |
| landing-page-design | **B3** Nested radius formula | Cards 16 px with 8 px inner gap give 8 px inner elements; pills are full radius |
| landing-page-design | **B4** Flat backgrounds, full borders only | Flat `#FAFAFA` page, white cards, borders on all four sides. The dotted texture is an SVG pattern, not a gradient |
| landing-page-design | **B5** Hero heading gradient (light theme `#000000` to `#666666`), max width 680 px | `.hero h1` uses the text-only gradient; heading and subheading capped at 680 px |
| landing-page-design | **B6** Icons from Phosphor, Solar or Iconamoon (never Material) | All icons are inline Phosphor SVGs (`src/filingsight/icons.py`); Streamlit emoji and Material icons are not used for UI chrome |
| landing-page-design | **B7** Motion: `cubic-bezier(0.22,1,0.36,1)`, 500 ms layout, at least 200 ms colour, transform and opacity only, reduced motion | `--ease` token on every transition; hover lifts use `transform`; `prefers-reduced-motion` turns animation off |
| landing-page-design | **B7** Navigation: compact floating bar, translucent with blur | `.topbar`: sticky pill, `backdrop-filter: blur(16px)`, status pill and GitHub link |
| landing-page-design | **B7** Scroll reveals on the compositor, no scroll listeners | KPI and answer cards rise 32 px and fade in; tagline uses CSS `animation-timeline: view()` |
| landing-page-design | **B9** States: hover, active, focus ring, skeleton loading, empty, inline errors | Buttons have hover, `scale(0.98)` active and a visible focus ring; a skeleton card replaces the spinner while answering; composed empty state with sample questions; inline refusal and fallback notices |
| landing-page-design | **B10** Ship checks: footer links, alt text, semantic HTML, 390 px with no overflow | `<nav>`, `<section>`, `<footer>`; `aria-hidden` on decoration; checked at 390 px wide (no horizontal scroll) |
| landing-page-design | **B11** Tagline reveal section, two or more lines, words light up one at a time | "Retrieval decided quality, not the model..." near the bottom; each word goes from 30% to full opacity |
| figma-designer | Two-layer colour tokens: Core Colors, then a Color System that aliases them, light mode mirrored from the dark reference | `:root` defines `--grayscale-*`, `--primary-*` (Core) and `--surface-*`, `--border-*`, `--text-*` (Color System) |
| figma-designer | Spacing and radius token scales | Same values as B2 and B3 above |

Decorative finance symbols (charts, dollar, percent, bank, receipt) and three blurred colour glows (indigo, pink, amber) sit behind the content in a fixed decoration layer, matching the Team 24 DealRoom app. Card surfaces stay flat white; depth comes from a two-step shadow token (`--shadow`, `--shadow-lg`), the hero is one white card with a 4 px brand stripe, tabs are a segmented pill, and Uber, Lyft and cited sources get a 3 px coloured left edge. The symbols float at about 10% opacity. They are `aria-hidden`, ignore pointer events, and stop moving under reduced motion.
