SHELL := /bin/bash

# Bundler installs to --user-install if not run with sudo, which lands in
# ~/.gem/ruby/<RUBY_VERSION>/bin.  Make sure that's on PATH so `bundle`
# is callable even from a fresh shell that didn't source a profile.
RUBY_VERSION := $(shell ruby -e 'puts RUBY_VERSION' 2>/dev/null)
USER_GEM_BIN := $(HOME)/.gem/ruby/$(RUBY_VERSION)/bin
export PATH   := $(USER_GEM_BIN):$(PATH)

HOST ?= 127.0.0.1
PORT ?= 4000

.PHONY: help install serve serve-prod build clean new publish resolve-cites

help:
	@echo "Blog targets:"
	@echo ""
	@echo "  make serve                  local preview with drafts → http://$(HOST):$(PORT)/blog/"
	@echo "  make serve-prod             local preview without drafts (matches live)"
	@echo "  make build                  one-shot production build into _site/"
	@echo "  make clean                  remove _site/, .jekyll-cache/, .sass-cache/"
	@echo "  make install                bundle install (run once or after Gemfile changes)"
	@echo ""
	@echo "  make new title='My Post'    create _drafts/YYYY-MM-DD-my-post.md"
	@echo "  make publish file=NAME.md   move _drafts/NAME.md → _posts/<today>-NAME.md"
	@echo "  make resolve-cites          fill url={...} for new bib entries (arxiv + Semantic Scholar)"
	@echo ""

resolve-cites:
	@command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
	@python3 -c 'import bibtexparser' 2>/dev/null \
	  || { echo "installing tools/requirements.txt..."; python3 -m pip install --user -r tools/requirements.txt; }
	python3 tools/resolve_urls.py

install:
	bundle config set --local path 'vendor/bundle'
	bundle install

serve:
	bundle exec jekyll serve --drafts --host $(HOST) --port $(PORT) --livereload

serve-prod:
	JEKYLL_ENV=production bundle exec jekyll serve --host $(HOST) --port $(PORT)

build:
	JEKYLL_ENV=production bundle exec jekyll build

clean:
	rm -rf _site .jekyll-cache .sass-cache

new:
	@if [ -z "$(title)" ]; then \
	  echo "usage: make new title='My Post Title'"; exit 1; \
	fi
	@slug=$$(echo "$(title)" \
	  | tr '[:upper:]' '[:lower:]' \
	  | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$$//g'); \
	date=$$(date +%Y-%m-%d); \
	file="_drafts/$$date-$$slug.md"; \
	if [ -e "$$file" ]; then echo "already exists: $$file"; exit 1; fi; \
	sed -E "s/^title:.*/title: $(title)/" _drafts/_template.md > "$$file"; \
	echo "created $$file"

publish:
	@if [ -z "$(file)" ]; then \
	  echo "usage: make publish file=YYYY-MM-DD-my-post.md"; exit 1; \
	fi
	@src="_drafts/$(file)"; \
	if [ ! -e "$$src" ]; then echo "no such draft: $$src"; exit 1; fi; \
	stripped=$$(echo "$(file)" | sed -E 's/^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}-//'); \
	date=$$(date +%Y-%m-%d); \
	dest="_posts/$$date-$$stripped"; \
	git mv "$$src" "$$dest"; \
	echo "moved $$src → $$dest"
