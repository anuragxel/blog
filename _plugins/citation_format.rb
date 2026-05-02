# IEEE CSL emits bibliography entries as "[N]Author..." with no space after
# the bracket.  Inject one for readability.

Jekyll::Hooks.register [:posts, :pages, :documents], :post_render do |doc|
  next unless doc.output
  doc.output = doc.output.gsub(/(\[\d+\])([A-Za-z])/, '\1 \2')
end
