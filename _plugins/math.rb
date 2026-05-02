require "base64"

# Wrap LaTeX math in HTML elements *before* kramdown runs.
#
# Kramdown's GFM mode does not protect inline $...$ from markdown — and
# even content placed inside HTML spans gets processed (so escapes like
# \{ are eaten and stray underscores can trigger emphasis).  To stop
# *anything* from touching the LaTeX, we stash it base64-encoded in a
# data attribute.  Client-side KaTeX (see _includes/head.html) decodes
# and renders.
#
# We swap display $$...$$ before inline $...$ so delimiters can't collide.

module BlogMathProtect
  DISPLAY_RE = /\$\$(.+?)\$\$/m
  # Inline: a single $ not preceded by another $ or backslash, with no
  # surrounding whitespace inside, and no newline in the body.
  INLINE_RE  = /(?<![\\$])\$(?!\s)([^\$\n]+?)(?<!\s)\$(?!\$)/

  def self.encode(tex)
    Base64.strict_encode64(tex.to_s)
  end

  def self.process(content)
    content
      .gsub(DISPLAY_RE) { %(<div class="math math-display" data-tex="#{encode(Regexp.last_match(1).strip)}"></div>) }
      .gsub(INLINE_RE)  { %(<span class="math math-inline" data-tex="#{encode(Regexp.last_match(1))}"></span>) }
  end
end

Jekyll::Hooks.register [:posts, :pages, :documents], :pre_render do |doc|
  next unless doc.extname.match?(/\.(md|markdown)\z/i)
  doc.content = BlogMathProtect.process(doc.content)
end
