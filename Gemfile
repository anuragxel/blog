source "https://rubygems.org"

gem "jekyll", "~> 4.3.4"

group :jekyll_plugins do
  gem "jekyll-sitemap",  "~> 1.4"
  gem "jekyll-seo-tag",  "~> 2.8"
  gem "jekyll-scholar",  "~> 7.1"
end

# Performance booster for watching directories on Linux
gem "listen", "~> 3.9"

# Required for Ruby 3.0+ (Net::HTTPResponse / webrick removed from stdlib)
gem "webrick", "~> 1.8"

# jekyll-scholar's transitive deps pull forwardable 1.4.0, which uses
# Ruby 3+ syntax and breaks Ruby 2.7's stdlib csv loading.  Pin older.
gem "forwardable", "~> 1.3.3"
