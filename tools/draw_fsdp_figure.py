"""Draw the FSDP concept figure with the blog's SVG/Cairo helpers.

Run: /usr/bin/python3 tools/draw_fsdp_figure.py
"""
from pathlib import Path
import draw_transformer_figures as draw

INK, BLUE, GREEN, FAINT = draw.INK, draw.BLUE, draw.GREEN, draw.FAINT
CENTERS = (190, 450)
COLORS = (BLUE, GREEN)


def rect(s, x, y, w, h, color, fill='white', dashed=False):
    dash = ' stroke-dasharray="5 5"' if dashed else ''
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{color}" stroke-width="1.8"{dash}/>')


def blocks(s, center, y, labels, owner=None, dashed=False):
    for i, (label, color) in enumerate(zip(labels, COLORS)):
        if owner is not None and i != owner:
            continue
        x=center-90+i*90
        rect(s,x,y,86,36,color,dashed=dashed)
        draw.text(s,x+43,y+25,label,color,19)


def down(s, x, y1, y2):
    draw.path(s,f'M{x} {y1} L{x} {y2} M{x-5} {y2-7} L{x} {y2} L{x+5} {y2-7}',INK,1.6)


def main():
    draw.OUT=Path(__file__).resolve().parents[1]/'assets/images/scaling'
    s=draw.start('FSDP temporarily assembles a layer',
        'Two device memories initially hold one colored parameter shard each. All-gather copies the shards so both briefly hold the full layer. Releasing the temporary copies leaves the original shards.', 515)
    draw.text(s,320,36,'FSDP',size=29)
    for c,name in zip((170,470),('Device 0','Device 1')):
        draw.text(s,c,77,name,size=23)
    for row,y in enumerate((96,246,396)):
        for owner,c in enumerate((170,470)):
            rect(s,c-94,y,188,102,FAINT)
            for i,color in enumerate(COLORS):
                if row==1 or owner==i:
                    x=c-79+i*80
                    rect(s,x,y+16,76,70,color,fill=('#eeeeff','#eaf5ee')[i])
                    draw.text(s,x+38,y+61,('W₀','W₁')[i],color,28)
    # Color follows a parameter slice when it is copied to the other device.
    draw.path(s,'M170 202 L470 241 M462 232 L470 241 L459 244',BLUE,1.8)
    draw.path(s,'M470 202 L170 241 M181 244 L170 241 L178 232',GREEN,1.8)
    draw.text(s,29,231,'all-gather',size=20,anchor='start')
    for c in (170,470):
        down(s,c,351,389)
    draw.text(s,320,367,'keep own',size=18)
    draw.text(s,320,387,'shard',size=18)
    draw.text(s,320,285,'materialized',size=15)
    draw.text(s,320,306,'for forward',size=15)
    draw.text(s,320,327,'pass',size=15)
    draw.save(s,'scaling1-fsdp.png')


if __name__=='__main__':
    main()
