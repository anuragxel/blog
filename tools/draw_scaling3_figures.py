"""Draw a pictorial array-placement comparison in the shared thin-ink style.

Run with /usr/bin/python3 tools/draw_scaling3_figures.py.
Only a 2x PNG is saved. The SVG representation stays in memory.
"""

from pathlib import Path

import draw_transformer_figures as draw
from draw_transformer_figures import BLUE, GREEN, INK, FAINT, path, start, text


def scene(s):
    """A single recognizable picture, reused exactly in every placement."""
    # Sky: cloud on the left, sun on the right.
    path(s, 'M9 27 Q3 18 15 17 Q16 7 28 12 Q37 6 43 18 Q53 20 47 28 Z', BLUE, 1.2)
    s.append(f'<circle cx="76" cy="22" r="10" fill="none" stroke="{BLUE}" stroke-width="1.2"/>')
    for d in ['M76 5 V1', 'M76 39 V43', 'M59 22 H55', 'M93 22 H97',
              'M64 10 L61 7', 'M88 34 L91 37', 'M88 10 L91 7', 'M64 34 L61 37']:
        path(s, d, BLUE, 1.2)
    # Land: a tree, a house, and a shared ground line.
    path(s, 'M1 91 Q26 86 49 91 Q76 97 99 89', GREEN, 1.2)
    path(s, 'M23 89 V69 M15 74 L23 79 L30 72', GREEN, 1.2)
    path(s, 'M10 70 Q4 57 14 54 Q9 40 23 40 Q34 34 39 47 Q48 50 40 60 Q45 73 31 73 Q21 79 10 70 Z', GREEN, 1.2)
    path(s, 'M58 90 V67 L74 52 L92 67 V92 M54 68 L74 49 L96 68', BLUE, 1.2)
    path(s, 'M69 91 V76 H80 V92 M83 69 H89 V76 H83 Z', BLUE, 1.2)


def picture(s, x, y, side, name, quadrant=None):
    s.append(f'<defs><clipPath id="{name}"><rect x="{x}" y="{y}" width="{side}" height="{side}"/></clipPath></defs>')
    s.append(f'<g clip-path="url(#{name})">')
    scale = side / 100 * (2 if quadrant is not None else 1)
    row, col = quadrant if quadrant is not None else (0, 0)
    s.append(f'<g transform="translate({x-col*side} {y-row*side}) scale({scale})">')
    scene(s)
    s.append('</g></g>')


def device(s, x, y):
    path(s, f'M{x} {y} H{x+106} V{y+106} H{x} Z', INK, 1.4)
    for delta in (19, 40, 65, 86):
        path(s, f'M{x+delta} {y-5} V{y} M{x+delta} {y+106} V{y+111} '
             f'M{x-5} {y+delta} H{x} M{x+106} {y+delta} H{x+111}', FAINT, 1.1)


def placements():
    s = start('One array, copied or divided across devices',
              'A picture containing a cloud, sun, tree and house is placed on four devices. '
              'Replication gives every device the whole picture. Sharding gives each device '
              'a distinct quarter, together making the same picture.', 490)
    text(s, 320, 30, 'One array', size=25)
    picture(s, 270, 48, 100, 'source')
    path(s, 'M270 48 H370 V148 H270 Z', FAINT, 1)
    path(s, 'M320 48 V148 M270 98 H370', FAINT, .8, dash=True)
    path(s, 'M300 160 Q226 174 169 195 M172 187 L169 195 L178 197', INK, 1.5)
    path(s, 'M340 160 Q414 174 471 195 M462 197 L471 195 L468 187', INK, 1.5)
    text(s, 169, 226, 'Replicated', BLUE, 25)
    text(s, 471, 226, 'Sharded', GREEN, 25)
    for i in range(4):
        row, col = divmod(i, 2)
        y = 247 + row*121
        # Left devices each contain a complete copy. Right devices hold quarters.
        for base, sharded in [(56, False), (358, True)]:
            x = base + col*121
            device(s, x, y)
            picture(s, x+5, y+5, 96, f'copy-{base}-{i}', (row, col) if sharded else None)
    draw.save(s, 'scaling2-placements.png')


if __name__ == '__main__':
    draw.OUT = Path(__file__).resolve().parents[1] / 'assets/images/scaling'
    placements()
