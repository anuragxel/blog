# Blog — anuragxel.github.io/blog

Personal blog for [Anurag Ghosh](https://anuragxel.github.io/). Built with Jekyll 4 and deployed via GitHub Actions.

## Local development

```bash
bundle install
bundle exec jekyll serve --livereload
```

Site renders at <http://localhost:4000/blog/>.

## Adding a post

Drop a markdown file in `_posts/` named `YYYY-MM-DD-title.md`:

```markdown
---
layout: post
title: My new post
date: 2026-05-02
description: Short description for the index card and SEO.
---

Body in Markdown. Math via `$inline$` or `$$display$$` (KaTeX).
Code in fenced blocks with language hints renders via Rouge.
```

## Deployment

Pushing to `gh-pages` triggers `.github/workflows/pages.yml`, which builds with Jekyll 4 and deploys via the official GitHub Pages Actions.

**One-time setup:** in repo settings → *Pages*, set **Source** to **GitHub Actions** (not "Deploy from a branch").

## Structure

- `_posts/` — post markdown
- `_layouts/` — page templates
- `_includes/` — head, header, footer partials
- `_sass/` — `_base.scss` (theme), `_syntax.scss` (code highlighting)
- `_plugins/math.rb` — protects LaTeX from kramdown via base64 data-tex
- `assets/css/main.scss` — entry point
- `_config.yml` — site config
