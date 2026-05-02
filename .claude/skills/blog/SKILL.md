---
name: blog
description: Use this skill for any work on the Jekyll blog at anuragxel.github.io/blog — creating new posts, previewing locally, publishing drafts, editing the theme, or pushing changes to the live site. Trigger phrases include "new post about X", "draft a post", "preview the blog", "publish that draft", "ship the blog post", "what's in my drafts", or any direct reference to blog tooling, posts, or theme.
---

# Blog iteration

Jekyll 4 blog deployed via GitHub Actions to <https://anuragxel.github.io/blog>.
The repo lives on the `gh-pages` branch; pushing triggers `.github/workflows/pages.yml`.

## Use the Makefile — don't reinvent commands

Every workflow is wired up as a `make` target. Run `make help` to list them. Prefer
these over typing `bundle exec jekyll ...` directly.

| Target | What it does |
|---|---|
| `make serve` | Local dev with drafts at <http://127.0.0.1:4000/blog/> (livereload on) |
| `make serve-prod` | Local dev without drafts — matches what the deployed site shows |
| `make build` | Production build into `_site/` |
| `make clean` | Remove `_site/`, `.jekyll-cache/`, `.sass-cache/` |
| `make install` | `bundle install` (only after Gemfile changes) |
| `make new title="My Post"` | Create `_drafts/YYYY-MM-DD-my-post.md` from `_drafts/_template.md` |
| `make publish file=NAME.md` | Move `_drafts/NAME.md` → `_posts/<today>-NAME.md` (uses `git mv`) |

## Common requests → action

- **"New post about X"** → `make new title="X"`, then open the new draft file and start writing in it. Don't push yet.
- **"Preview the blog"** → run `make serve` in the background; then either `Read` the generated `_site/index.html` to verify, or remind the user to tunnel `ssh -L 4000:127.0.0.1:4000 <host>` if they're on a remote node.
- **"Publish [draft name]"** → `make publish file=…`, then `git commit` with a meaningful message and `git push origin gh-pages`. The Action deploys.
- **"Ship the typo fix" / "deploy"** → `git add -A && git commit -m "<terse why>" && git push origin gh-pages`. CI handles the rest.
- **"Make the blog look like X"** → edits go in `_sass/_base.scss` (theme) or `_includes/head.html` (head), then `make build` and inspect `_site/...` to verify.

## Authoring conventions

- **Math:** `$inline$` and `$$display$$`. `_plugins/math.rb` base64-stashes math
  in `data-tex` attributes before kramdown runs, so backslash-braces and stray
  underscores survive untouched. KaTeX renders client-side from `_includes/head.html`.
- **Code:** triple-fenced blocks with a language hint render via Rouge. Light theme
  in `_sass/_syntax.scss`.
- **Citations:** `jekyll-scholar` with IEEE numeric style. Shared bib at
  `_bibliography/references.bib`. Inline: `{% raw %}{% cite key1 key2 %}{% endraw %}`.
  References at end of post: `## References` then `{% raw %}{% bibliography --cited %}{% endraw %}`.
  Adding new refs = paste BibTeX into `references.bib`; abbreviated `booktitle={CVPR}`
  is preserved as-is. `_plugins/citation_format.rb` cleans up the IEEE CSL's
  missing space after `[N]`.
- **Clickable titles:** entries with a `url={...}` field render their title as a
  link in the bibliography (`_plugins/citation_links.rb` handles wrapping and
  strips IEEE's `Available at:` duplicate). To fill URLs in bulk:
  `make resolve-cites` runs `tools/resolve_urls.py` (arXiv ID detection +
  Semantic Scholar lookup). Manual `url=` fields always win — paste those for
  project pages.
- **Frontmatter:** `layout: post`, `title:`, optional `description:` (used in index
  card and `og:description`). Date comes from the filename (`YYYY-MM-DD-slug.md`).

## Don't

- Don't run `bundle update` casually — the locked Jekyll 4.3 / kramdown / rouge
  combo is what the math plugin and theme were tested against. Update intentionally,
  not as a side effect.
- Don't reintroduce RSS/jekyll-feed — explicitly removed by user preference.
- Don't append `Co-Authored-By: Claude` to commits in this repo.
- Don't add tags/categories/dark-mode/reading-time on speculation — wait until
  there's a concrete reason and a clear place to put them.
