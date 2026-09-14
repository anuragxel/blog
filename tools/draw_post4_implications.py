"""Small diagrams for Post 4; use the shared drawing primitives to save PNGs.

    /usr/bin/python3 tools/draw_post4_implications.py
"""

import sys

sys.dont_write_bytecode = True

from draw_transformer_figures import INK, BLUE, GREEN, start, text, path, arrow, matrix, save


def box(s, x, y, width, height, lines, color=INK, size=19):
    path(s, f'M{x} {y} Q{x+width/2} {y-2} {x+width} {y} L{x+width} {y+height} Q{x+width/2} {y+height-1} {x} {y+height} Z', color)
    for i, line in enumerate(lines):
        text(s, x+width/2, y+height/2+(i-(len(lines)-1)/2)*25+7, line, color, size)


def down(s, x, y1, y2, color=INK):
    path(s, f'M{x} {y1} V{y2} M{x-5} {y2-8} L{x} {y2} L{x+5} {y2-8}', color)


def rag():
    s = start('RAG adds references to the context', 'The original context and retrieved document tokens combine into input X. Self-attention processes this expanded context with the same projection weights. Context and document tokens have the same width d; concatenation grows the row count from N to N plus M.', 300)
    text(s, 93, 37, 'context', BLUE)
    matrix(s, 54, 50, 78, 55, BLUE)
    text(s, 93, 130, 'N × d', BLUE, 18)
    text(s, 93, 153, 'doc tokens', GREEN, 19)
    matrix(s, 54, 166, 78, 55, GREEN)
    text(s, 93, 246, 'M × d', GREEN, 18)
    path(s, 'M145 78 H175 V137 M145 193 H175 V137', INK)
    arrow(s, 175, 137, 211, 137)
    text(s, 257, 40, 'X', size=26)
    text(s, 257, 246, '(N + M) × d', size=18)
    # Concatenation preserves the three context rows and three document rows.
    path(s, 'M233 59 H226 V214 H233 M281 59 H288 V214 H281', INK)
    for y, color in [(75, BLUE), (99, BLUE), (123, BLUE), (150, GREEN), (174, GREEN), (198, GREEN)]:
        path(s, f'M239 {y} H275', color, 1.8)
    arrow(s, 301, 137, 340, 137)
    box(s, 352, 106, 153, 64, ['self-attention'])
    text(s, 428, 204, 'same weights', size=18)
    arrow(s, 519, 137, 557, 137)
    text(s, 596, 144, 'out', GREEN)
    text(s, 399, 282, 'more rows, same token width', size=19)
    save(s, 'rag-context.png')


def kv_cache():
    s = start('The cache grows while its vector types stay compatible',
              'At one attention head, cached keys have shape (t minus 1) by d k and cached values have shape (t minus 1) by d v. Append a key of width d k and a value of width d v. The current query has width d k; the lookup returns a value vector of width d v.', 355)
    for x, name, dim, color in [(226, 'keys K', 'dₖ', BLUE), (398, 'values V', 'dᵥ', GREEN)]:
        text(s, x+38, 27, name, color)
        text(s, x+38, 53, '(t − 1) × '+dim, color, 17)
        matrix(s, x, 70, 76, 75, color)
        text(s, x+38, 179, 'kₜ : dₖ' if name.startswith('keys') else 'vₜ : dᵥ', color, 18)
        path(s, f'M{x-7} 159 V189 H{x} M{x+83} 159 V189 H{x+76}', color)
        down(s, x+38, 202, 235, color)
    text(s, 90, 102, 'reuse past rows', size=18)
    text(s, 90, 180, 'append one row', size=18)
    box(s, 231, 247, 213, 60, ['attention lookup'])
    text(s, 73, 243, 'query', BLUE, 19)
    text(s, 73, 285, 'qₜ', BLUE, 28)
    text(s, 73, 325, 'width dₖ', BLUE, 18)
    arrow(s, 110, 277, 218, 277, BLUE)
    arrow(s, 457, 277, 539, 277, GREEN)
    text(s, 578, 285, 'oₜ', GREEN, 28)
    text(s, 578, 325, 'width dᵥ', GREEN, 18)
    save(s, 'kv-cache.png')


def tools():
    s = start('Tool interfaces are supplied in the context', 'A request and tool descriptions enter the language model as context. The model uses its learned tool-use behavior to generate a tool name and arguments. Its weights stay fixed at inference.', 260)
    text(s, 103, 35, 'request', BLUE)
    matrix(s,  60, 49, 86, 49, BLUE)
    text(s, 103, 146, 'tool descriptions', GREEN, 19)
    matrix(s, 60, 160, 86, 49, GREEN)
    path(s, 'M159 74 H195 V127 M159 184 H195 V127')
    arrow(s, 195, 127, 248, 127)
    box(s, 260, 93, 158, 68, ['language model'])
    text(s, 339, 194, 'fixed weights', size=18)
    arrow(s, 431, 127, 477, 127)
    text(s, 554, 103, 'tool call', GREEN)
    text(s, 554, 141, 'name, args', GREEN, 20)
    save(s, 'tool-context.png')


def reuse():
    s = start('Layer reuse preserves the token interface',
              'A transformer block accepts and returns N by d tokens of conceptual Type Concept. The output feeds back into the same block with shared weights. Matching the interface makes the loop well-defined; training must make its repeated updates useful.', 270)
    text(s, 91, 35, 'Type[Concept]', BLUE, 21)
    matrix(s, 53, 58, 76, 90, BLUE)
    text(s, 91, 177, 'N × d', BLUE, 21)
    arrow(s, 143, 103, 223, 103)
    box(s, 237, 68, 166, 70, ['transformer block', 'attention + MLP'], size=18)
    text(s, 320, 35, 'same weights', size=19)
    arrow(s, 417, 103, 496, 103, GREEN)
    text(s, 549, 35, 'Type[Concept]', GREEN, 21)
    matrix(s, 511, 58, 76, 90, GREEN)
    text(s, 549, 177, 'N × d', GREEN, 21)
    path(s, 'M467 104 V224 Q467 239 452 239 H185 Q170 239 170 224 V104', GREEN)
    path(s, 'M165 112 L170 104 L175 112', GREEN)
    text(s, 320, 216, 'output fits the input', GREEN, 21)
    save(s, 'layer-reuse.png')


def multimodal():
    s = start('The adapter connects visual and language token interfaces',
              'A vision encoder produces Z with shape N v by d v and conceptual Type Visual. The learned adapter W A maps Z to H v with shape N v by d and conceptual Type LinguoVisual. Text tokens of Type Language have width d too. The language model accepts both kinds after alignment training. These are conceptual types, not formal type guarantees.', 390)
    text(s, 72, 40, 'image', BLUE, 20)
    down(s, 72, 51, 85, BLUE)
    box(s, 20, 98, 104, 64, ['vision', 'encoder'])
    arrow(s, 137, 130, 162, 130)
    text(s, 201, 83, 'Z', BLUE, 25)
    matrix(s, 177, 105, 48, 52, BLUE)
    text(s, 201, 188, 'Nᵥ × dᵥ', BLUE, 18)
    text(s, 201, 214, 'Visual', BLUE, 20)
    arrow(s, 239, 130, 271, 130)
    box(s, 284, 102,  60, 56, ['W_A'])
    text(s, 314, 83, 'adapter', size=18)
    arrow(s, 358, 130, 390, 130, GREEN)
    text(s, 435, 83, 'Hᵥ', GREEN, 25)
    matrix(s, 409, 105, 52, 52, GREEN)
    text(s, 435, 188, 'Nᵥ × d', GREEN, 18)
    text(s, 435, 214, 'LinguoVisual', GREEN, 19)
    arrow(s, 475, 130, 544, 130, GREEN)
    box(s, 556, 102,  60, 56, ['LLM'])
    text(s, 306, 270, 'text tokens', BLUE, 19)
    text(s, 306, 300, 'Language', BLUE, 20)
    matrix(s, 409, 252, 52, 50, BLUE)
    text(s, 435, 327, 'Nₜ × d', BLUE, 18)
    path(s, 'M475 277 H524 V143', BLUE)
    path(s, 'M519 151 L524 143 L529 151', BLUE)
    text(s, 320, 363, 'LLM input: Language | LinguoVisual', size=20)
    save(s, 'multimodal-adapter.png')


if __name__ == '__main__':
    for draw in (rag, kv_cache, tools, reuse, multimodal):
        draw()
