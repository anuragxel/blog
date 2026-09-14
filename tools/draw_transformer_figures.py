"""Editable source for the blog diagrams. Saves only 2x PNG assets.

    /usr/bin/python3 tools/draw_transformer_figures.py

Uses the system Python packages PyGObject, librsvg, and Cairo.

Blue circles denote reference points/keys; green tags denote labels/values.
The two aligned flows map k-NN terminology to attention terminology.
Point positions are schematic: attention weights come from learned scores.
"""

from html import escape
from pathlib import Path

import cairo
import gi

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg

OUT = Path(__file__).resolve().parents[1] / 'assets/images/transformer'
INK, BLUE, GREEN, FAINT = '#252525', '#3538b5', '#17803d', '#b6b6bc'


def start(title, description, height):
    return [f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}" viewBox="0 0 640 {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title>
<desc id="desc">{escape(description)}</desc>
<rect width="640" height="{height}" fill="white"/>
<g font-family="'Comic Sans MS', Chilanka, cursive" font-size="21" fill="{INK}" stroke-linecap="round" stroke-linejoin="round">''']


def text(s, x, y, label, color=INK, size=21, anchor='middle'):
    label = escape(label).replace('W_O', 'W<tspan baseline-shift="sub" font-size="70%">O</tspan>').replace('W_A', 'W<tspan baseline-shift="sub" font-size="70%">A</tspan>').replace('W_Q', 'W<tspan baseline-shift="sub" font-size="70%">Q</tspan>').replace('W_K', 'W<tspan baseline-shift="sub" font-size="70%">K</tspan>').replace('W_V', 'W<tspan baseline-shift="sub" font-size="70%">V</tspan>')
    s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="{size}">{label}</text>')


def path(s, d, color=INK, width=2, dash=False):
    s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"' + (' stroke-dasharray="5 6"' if dash else '') + '/>')


def arrow(s, x1, y1, x2, y2, color=INK, width=2):
    # Explicit arrowheads render consistently in both SVG and librsvg PNGs.
    path(s, f'M{x1} {y1} Q{(x1+x2)/2} {y1-1} {x2} {y2}', color, width)
    path(s, f'M{x2-8} {y2-5} L{x2} {y2} L{x2-8} {y2+5}', color, 2)


def dot(s, x, y, color=BLUE):
    s.append(f'<circle cx="{x}" cy="{y}" r="7" fill="white" stroke="{color}" stroke-width="2.5"/>')


def tag(s, x, y, label, color=GREEN):
    path(s, f'M{x-17} {y-16} Q{x} {y-18} {x+17} {y-15} L{x+16} {y+16} Q{x} {y+15} {x-17} {y+17} Z', color)
    text(s, x, y+7, str(label), color, 22)


def pair(s, x, y, label, color=GREEN):
    path(s, f'M{x+8} {y} L{x+26} {y}', FAINT, 1.5)
    dot(s, x, y)
    tag(s, x+44, y, label, color)


def save(s, name):
    OUT.mkdir(parents=True, exist_ok=True)
    # SVG is an in-memory drawing representation; only PNG files are saved.
    drawing = ('\n'.join(s + ['</g></svg>'])+'\n').encode()
    handle = Rsvg.Handle.new_from_data(drawing)
    dimensions = handle.get_dimensions()
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                 dimensions.width * 2, dimensions.height * 2)
    context = cairo.Context(surface)
    context.scale(2, 2)
    handle.render_cairo(context)
    surface.write_to_png(str(OUT/name))


def lookup():
    s = start('k-NN and self-attention: corresponding lookup flows',
              'Two aligned left-to-right flows. In k-NN, a test point selects neighbors from a reference set and their labels produce a prediction. In self-attention, a query scores keys and the corresponding values contribute to an output with soft weights. Circles represent references or keys; square tags represent their paired labels or values.', 580)
    for offset, soft in [(0, False), (290, True)]:
        s.append(f'<g transform="translate(0 {offset})">')
        text(s, 24, 32, 'Self-attention' if soft else 'k-NN', size=26, anchor='start')
        for x, label, color in [(64, 'query' if soft else 'test point', BLUE),
                                 (228, 'keys' if soft else 'reference set', BLUE),
                                 (368, 'values' if soft else 'labels', GREEN),
                                 (563, 'output' if soft else 'prediction', GREEN)]:
            text(s, x,  80, label, color, 20)
        widths = [1, 4, 2.6, .7] if soft else [0, 2.5, 2.5, 0]
        for j, (y, subscript) in enumerate(zip([112, 154, 196, 238], ['₁', '₂', '₃', 'ₙ'])):
            active = widths[j] > 0
            if active:
                path(s, f'M87 175 Q151 {y} 218 {y}', BLUE, widths[j])
                path(s, f'M389 {y} Q457 {y} 517 175', GREEN, widths[j])
            path(s, f'M239 {y} L347 {y}', FAINT, 1.4, dash=True)
            dot(s, 228, y, BLUE if active else FAINT)
            tag(s, 368, y, ('v' if soft else 'y')+subscript, GREEN if active else FAINT)
        text(s, 64, 183, 'qᵢ' if soft else 'x', BLUE, 29)
        arrow(s, 517, 175, 541, 175, GREEN)
        text(s, 568, 184, 'oᵢ' if soft else 'ŷ', GREEN, 29)
        text(s, 155, 274, 'soft weighting' if soft else 'select neighbors', BLUE, 18)
        text(s, 461, 274, 'oᵢ = Σⱼ αᵢⱼ vⱼ' if soft else 'vote / average', GREEN, 21)
        s.append('</g>')
    path(s, 'M24 289 L616 289', '#dedee2', 1)
    save(s, 'soft-knn.png')


def matrix(s, x, y, width, height, color):
    """Bracketed rows represent token vectors, not individual scalar entries."""
    path(s, f'M{x+7} {y} H{x} V{y+height} H{x+7}', color, 2)
    path(s, f'M{x+width-7} {y} H{x+width} V{y+height} H{x+width-7}', color, 2)
    for fraction in [.2, .5, .8]:
        row_y = y + height * fraction
        path(s, f'M{x+14} {row_y} H{x+width-14}', color, 1.6)


def weights():
    s = start('Self-attention: projection weights and projections',
              'The same input matrix X branches through three separate projection weight matrices W Q, W K, and W V, producing Q equals X W Q, K equals X W K, and V equals X W V. The weights are fixed at inference; the projections depend on X. Bracketed rows depict token vectors.', 390)
    text(s, 76, 31, 'input', size=23)
    text(s, 301, 31, 'projection weights', size=21)
    text(s, 531, 31, 'projections', size=23)
    text(s, 301, 60, 'fixed at inference', size=17)
    text(s, 531, 60, 'depend on X', size=17)
    text(s, 76, 148, 'X', size=29)
    matrix(s, 32, 166, 88, 108, INK)
    path(s, 'M129 220 H179 M179 120 V320', INK, 2)
    for y, name, color in [(120, 'Q', BLUE), (220, 'K', BLUE), (320, 'V', GREEN)]:
        arrow(s, 179, y, 251, y, INK)
        path(s, f'M264 {y-27} Q302 {y-29} 340 {y-27} L339 {y+28} Q301 {y+26} 264 {y+28} Z', color, 2)
        text(s, 302, y+9, 'W_'+name, color, 29)
        arrow(s, 352, y, 465, y, color)
        text(s, 533, y-30, name+' = X W_'+name, color, 23)
        matrix(s, 486, y-17, 94,  50, color)
    save(s, 'weights-vs-projections.png')


if __name__ == '__main__':
    lookup()
    weights()
