# Blog — anuragxel.github.io/blog

Personal blog for [Anurag Ghosh](https://anuragxel.github.io/). Built with Jekyll 4 and deployed via GitHub Actions.

## Local development

```bash
bundle install
bundle exec jekyll serve --livereload
```

Site renders at <http://localhost:4000/blog/>.

## Authoring

Two workflows depending on whether you want to preview locally or on the live site.

### Iterate on the deployed site
Easiest if local Ruby is annoying. Drop the new post directly in `_posts/`
with the current date in the filename (`YYYY-MM-DD-slug.md`), push, and
refresh the live URL. Keep committing follow-ups until happy.

### Local drafts
Drafts live in `_drafts/`. They show up under `bundle exec jekyll serve --drafts`
but are skipped in production builds — the deployed site never sees them.
A starter is at `_drafts/_template.md`. To publish, rename to
`YYYY-MM-DD-slug.md` and move into `_posts/`.

### Frontmatter

```markdown
---
layout: post
title: My new post
description: One-sentence summary for the index card and social previews.
---
```

Body is Markdown. Math via `$inline$` or `$$display$$` (KaTeX, via build-time plugin).
Fenced code blocks with language hints render via Rouge.

### Citations

One shared bib at `_bibliography/references.bib`. Cite inline and emit a
references list with [`jekyll-scholar`](https://github.com/inukshuk/jekyll-scholar):

```markdown
SfM has been studied for decades {% raw %}{% cite schonberger2016structure %}{% endraw %},
and place-recognition methods like NetVLAD {% raw %}{% cite arandjelovic2016netvlad torii2011visual %}{% endraw %}.

## References

{% raw %}{% bibliography --cited %}{% endraw %}
```

`--cited` emits only the entries actually referenced in the post (not the
whole bib). Style is IEEE numeric `[1]`. Drop new BibTeX entries straight
into `references.bib` — the abbreviated `booktitle={CVPR}` format renders
as-is, no expansion.

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
