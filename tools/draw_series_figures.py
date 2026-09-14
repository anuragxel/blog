"""Additional diagrams for Transformer Parts 1–3, saved as PNGs only.

    /usr/bin/python3 tools/draw_series_figures.py

The shared primitives keep labels, matrix rows, and colors consistent with Post 4.
"""

import sys
sys.dont_write_bytecode = True

from draw_transformer_figures import INK, BLUE, GREEN, FAINT, start, text, path, arrow, matrix, dot, save
from draw_post4_implications import box, down


def bottleneck():
    s = start('Addressable tokens versus a fixed recurrent state',
              'A transformer block maps an N by d token matrix to another N by d matrix. A recurrent update combines the previous state with the current token to produce another state of the same fixed width d h. Keeping token slots does not guarantee preservation of their information.', 435)
    text(s, 24,  30, 'Token slots preserved', size=24, anchor='start')
    matrix(s,  40,  60, 88, 112, BLUE)
    text(s, 84, 202, 'N × d', BLUE)
    arrow(s, 142, 116, 230, 116)
    box(s, 244,  80, 160, 72, ['transformer', 'block'])
    arrow(s, 418, 116, 505, 116, GREEN)
    matrix(s, 519, 60, 88, 112, GREEN)
    text(s, 563, 202, 'N × d', GREEN)
    path(s, 'M24 228 H616', '#dedee2', 1)
    text(s, 24, 265, 'Fixed-state recurrence', size=24, anchor='start')
    text(s,  70, 363, 'hₜ₋₁', BLUE, 29)
    text(s, 70, 400, 'width dₕ', BLUE, 19)
    arrow(s, 115, 354, 232, 354)
    box(s, 246, 328, 156, 52, ['state update'])
    text(s, 324, 290, 'xₜ', BLUE, 25)
    down(s, 324, 302, 317, BLUE)
    arrow(s, 416, 354, 532, 354, GREEN)
    text(s, 580, 363, 'hₜ', GREEN, 29)
    text(s, 580, 400, 'width dₕ', GREEN, 19)
    save(s, 'token-bottleneck.png')


def self_cross():
    s = start('Self-attention and cross-attention follow the query row count',
              'In self-attention, X supplies all three projections. In cross-attention, X supplies Q and Y supplies K and V through separate weight matrices. X has N rows and Y has M rows. In both cases, attention returns N rows, one per query.', 500)
    for offset, cross in [(0, False), (250, True)]:
        s.append(f'<g transform="translate(0 {offset})">')
        text(s, 24, 30, 'Cross-attention' if cross else 'Self-attention', size=24, anchor='start')
        if cross:
            text(s, 65, 83, 'X', BLUE, 28)
            text(s, 65, 109, 'N rows', BLUE, 17)
            arrow(s, 98, 75, 179, 75, BLUE)
            text(s, 65, 173, 'Y', GREEN, 28)
            text(s, 65, 201, 'M rows', GREEN, 17)
            path(s, 'M98 165 H142 M142 135 V195', GREEN)
            arrow(s, 142, 135, 179, 135, GREEN)
            arrow(s, 142, 195, 179, 195, GREEN)
        else:
            text(s, 65, 143, 'X', BLUE, 28)
            text(s, 65, 177, 'N rows', BLUE, 17)
            path(s, 'M98 135 H142 M142 75 V195', BLUE)
            for y in [75, 135, 195]:
                arrow(s, 142, y, 179, y, BLUE)
        for y, name, color in [(75, 'Q', BLUE), (135, 'K', BLUE), (195, 'V', GREEN)]:
            box(s, 191, y-21, 72, 42, ['W_'+name], color, size=23)
            arrow(s, 276, y, 311, y, color)
            text(s, 333, y+8, name, color, 25)
            arrow(s, 353, y, 389, y, color)
        box(s, 402, 55, 117, 160, ['attention'])
        arrow(s, 532, 135, 564, 135, GREEN)
        text(s, 598, 143, 'out', GREEN, 21)
        text(s, 590, 177, 'N rows', GREEN, 17)
        s.append('</g>')
    path(s, 'M24 247 H616', '#dedee2', 1)
    save(s, 'self-vs-cross.png')


def multihead():
    s = start('Parallel heads keep retrieved results separate before mixing',
              'One token x i branches to two head-specific soft lookups over the same input context. Head 1 and head 2 have separate learned similarities and produce o i 1 and o i 2. These outputs concatenate and then pass through the output projection W O.', 350)
    text(s, 75, 144, 'token', BLUE, 20)
    text(s, 75, 181, 'xᵢ', BLUE, 29)
    path(s, 'M108 172 H144 M144 93 V249', BLUE)
    for y, sub, color in [(93, '₁', BLUE), (249, '₂', GREEN)]:
        arrow(s, 144, y, 178, y, color)
        box(s, 190, y-33, 156, 66, ['head '+('1' if sub == '₁' else '2'), 'lookup in X'], color)
        text(s, 268, y-47, 'similarity M'+sub, color, 20)
        arrow(s, 359, y, 385, y, color)
        text(s, 415, y+8, 'oᵢ'+sub, color, 25)
        path(s, f'M444 {y} H469 V172', color)
    arrow(s, 469, 172, 484, 172)
    # A partitioned vector is the concatenation, not an average of the heads.
    path(s, 'M500 138 H493 V206 H500 M527 138 H534 V206 H527')
    path(s, 'M502 156 H525', BLUE, 3)
    path(s, 'M502 188 H525', GREEN, 3)
    text(s, 511, 124, 'concat', size=18)
    arrow(s, 543, 172, 560, 172)
    box(s, 571, 148, 50, 48, ['W_O'], size=23)
    text(s, 595, 231, 'mix', size=19)
    text(s, 320, 328, 'same context X, separate lookups', size=21)
    save(s, 'multi-head-lookups.png')


def score_matrix(s, x, y, entries):
    cell = 54
    text(s, x+27, y-15, 'A', size=20)
    text(s, x+81, y-15, 'B', size=20)
    text(s, x-22, y+35, 'A', size=20)
    text(s, x-22, y+89, 'B', size=20)
    path(s, f'M{x+5} {y-2} H{x-3} V{y+110} H{x+5} M{x+103} {y-2} H{x+111} V{y+110} H{x+103}')
    for row in range(2):
        for col in range(2):
            color = FAINT if row == col else (BLUE if row == 0 else GREEN)
            text(s, x+cell*col+27, y+cell*row+35, entries[row][col], color, 25)
    path(s, f'M{x+9} {y+9} L{x+99} {y+99}', FAINT, 1, dash=True)


def tied_scores():
    s = start('Tying query and key weights makes scores symmetric',
              'Two symbolic score matrices compare tied and separate query/key weights. Tied weights force the off-diagonal scores A to B and B to A to equal the same s. Separate weights allow scores a and b to differ. This compares unmasked dot-product scores before softmax, without additional positional terms.', 340)
    text(s, 163, 32, 'tied Q/K weights', size=23)
    text(s, 478, 32, 'separate weights', size=23)
    text(s, 163, 68, 'W_Q = W_K', size=23)
    text(s, 478, 68, 'W_Q ≠ W_K', size=23)
    score_matrix(s, 113, 117, [['·', 's'], ['s', '·']])
    score_matrix(s, 428, 117, [['·', 'a'], ['b', '·']])
    text(s, 163, 274, 'A → B = B → A', size=21)
    text(s, 478, 274, 'a and b can differ', size=21)
    text(s, 320, 321, 'query–key scores, before softmax', size=20)
    save(s, 'tied-qk-scores.png')


def convex():
    s = start('Softmax mixes values inside their convex hull',
              'Three value vectors form the vertices of a triangle. A weighted average with nonnegative weights summing to one lies inside their triangle. The diagram illustrates one head output before output projection and residual addition.', 325)
    pts = [(80, 249), (178, 62), (307, 249)]
    s.append('<path d="M80 249 L178 62 L307 249 Z" fill="#17803d" fill-opacity="0.06" stroke="#17803d" stroke-width="2"/>')
    for (x,y), label, dx, dy in zip(pts, ['v₁','v₂','v₃'], [-19,0,19], [28,-20,28]):
        dot(s,x,y,GREEN)
        text(s,x+dx,y+dy,label,GREEN,24)
    dot(s,180,187,BLUE)
    text(s,199,180,'o',BLUE,25)
    text(s,475,117,'o = Σⱼ αⱼ vⱼ',BLUE,25)
    text(s,475,168,'αⱼ ≥ 0',GREEN,23)
    text(s,475,208,'Σⱼ αⱼ = 1',GREEN,23)
    text(s,320,309,'the output stays among the values',size=21)
    save(s,'attention-convex-hull.png')


def array_attention():
    s = start('Pointer, storage, and output as a soft lookup',
              'Two aligned flows compare addressed memory with attention. A pointer p holding address a 2 selects storage contents v 2. Attention uses query q to score stored keys and blend their associated values. The requested address corresponds to the query, stored addresses to keys, and stored contents to values. Storage contains both the addressing scheme and its associated contents; the attention output is a weighted blend, not necessarily one stored value.', 650)
    for offset, soft in [(0, False), (325, True)]:
        s.append(f'<g transform="translate(0 {offset})">')
        text(s, 24, 32, 'Attention: soft lookup' if soft else 'Array / memory lookup', size=25, anchor='start')
        text(s, 76, 77, 'query' if soft else 'pointer', BLUE, 21)
        text(s, 305, 77, 'key–value memory' if soft else 'storage', size=20)
        text(s, 560, 77, 'output', GREEN, 21)
        path(s, 'M197 90 H412 V284 H197 Z', '#b6b6bc', 1.5)
        text(s, 244, 115, 'keys' if soft else 'addresses', BLUE, 18)
        text(s, 364, 115, 'values' if soft else 'contents', GREEN, 18)
        widths = [1.1, 4, 2.3] if soft else [0, 2.5, 0]
        for j, (y, sub) in enumerate(zip([145, 200, 255], ['₁', '₂', 'ₙ'])):
            active = widths[j] > 0
            color_k, color_v = (BLUE, GREEN) if active else (FAINT, FAINT)
            if active:
                path(s, f'M{100 if soft else 130} 200 Q163 {y} 217 {y}', BLUE, widths[j])
                path(s, f'M391 {y} Q469 {y} 520 200', GREEN, widths[j])
            box(s, 220, y-18, 48, 36, [('k' if soft else 'a')+sub], color_k, size=23)
            path(s, f'M280 {y} H329', FAINT, 1.4, dash=True)
            box(s, 340, y-18, 48, 36, ['v'+sub], color_v, size=23)
        arrow(s, 520, 200, 539, 200, GREEN)
        text(s, 76, 208, 'q' if soft else 'p = a₂', BLUE, 26)
        text(s, 560, 208, 'o' if soft else 'v₂', GREEN, 29)
        text(s, 160, 313, 'similarity → softmax' if soft else 'read at address p', BLUE, 19)
        text(s, 467, 313, 'o = Σⱼ αⱼ vⱼ' if soft else '*p = v₂', GREEN, 22)
        s.append('</g>')
    path(s, 'M24 323 H616', '#dedee2', 1)
    save(s, 'array-vs-attention.png')


if __name__ == '__main__':
    for draw in (bottleneck, self_cross, multihead, tied_scores, convex, array_attention):
        draw()
