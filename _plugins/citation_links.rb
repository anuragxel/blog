require "bibtex"

# When a bibliography is rendered (jekyll-scholar), wrap each entry's
# title with a link to the entry's `url` field.  Title detection:
#   - First curly-quoted phrase  →  "Title,"  (most paper entry types)
#   - First <i>...</i>            →  fallback (books, theses, etc.)

module BlogCitationLinks
  BIB_PATH = File.join(Dir.pwd, "_bibliography", "references.bib")

  def self.url_map
    return @url_map if defined?(@url_map)
    @url_map = {}
    return @url_map unless File.exist?(BIB_PATH)
    BibTeX.open(BIB_PATH).each do |e|
      next unless e.respond_to?(:url) && !e.url.to_s.empty?
      @url_map[e.key.to_s] = e.url.to_s
    end
    @url_map
  rescue StandardError => err
    Jekyll.logger.warn "citation_links:", "couldn't read #{BIB_PATH}: #{err}"
    @url_map = {}
  end

  # IEEE CSL appends ", Available: <url>" or ". [Online]. Available: <url>" when
  # an entry has a url field.  (Note: ieee.csl on l.61 redefines the
  # "available at" term to just "available", so the prefix is "Available:",
  # not "Available at:".)  We're using the url to link the title instead,
  # so strip that trailing fragment.
  AVAILABLE_AT_RE = /[,.]?\s*(?:\[[A-Za-z]+\]\.\s*)?Available(?:\s+at)?:\s*\S+\s*\.?\s*\z/

  def self.link_title(body, url)
    body = body.sub(AVAILABLE_AT_RE, "").rstrip
    body = body + "." unless body.end_with?(".", ">")
    return body if body.include?("</a>") # already linked
    # Curly-quoted title: U+201C ... U+201D
    if body =~ /“[^”]+”/
      return body.sub(/(“)([^”]+)(”)/) {
        %(#{Regexp.last_match(1)}<a href="#{url}">#{Regexp.last_match(2)}</a>#{Regexp.last_match(3)})
      }
    end
    # Italic title (books)
    if body =~ /<i>[^<]+<\/i>/
      return body.sub(/<i>([^<]+)<\/i>/) {
        %(<i><a href="#{url}">#{Regexp.last_match(1)}</a></i>)
      }
    end
    body
  end
end

Jekyll::Hooks.register [:posts, :pages, :documents], :post_render do |doc|
  next unless doc.output && doc.output.include?('class="bibliography"')
  next if BlogCitationLinks.url_map.empty?
  doc.output = doc.output.gsub(
    /<li><span id="([^"]+)">(.*?)<\/span><\/li>/m
  ) do
    key  = Regexp.last_match(1)
    body = Regexp.last_match(2)
    url  = BlogCitationLinks.url_map[key]
    if url
      %(<li><span id="#{key}">#{BlogCitationLinks.link_title(body, url)}</span></li>)
    else
      Regexp.last_match(0)
    end
  end
end
