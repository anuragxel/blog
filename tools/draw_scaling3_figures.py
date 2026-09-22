"""Draw the same arrays on four devices with three different mesh shapes.

Run /usr/bin/python3 tools/draw_scaling3_figures.py. Grids show scalars;
colors and small shard labels track pieces from the source to device memory.
"""
from pathlib import Path

import draw_transformer_figures as d

INK, BLUE, GREEN, FAINT = d.INK, d.BLUE, d.GREEN, d.FAINT
BATCH_COLORS = ('#087c9d', '#b85b12', '#b23c68', '#8a6c12')
WEIGHT_COLORS = (BLUE, GREEN)
FILLS = {
    INK: '#f0f0f2', BLUE: '#eeeeff', GREEN: '#eaf5ee',
    '#087c9d': '#e5f4f8', '#b85b12': '#fff0e3',
    '#b23c68': '#fbeaf1', '#8a6c12': '#faf3d9',
}


def box(s, x, y, w, h, color=FAINT):
    d.path(s, f'M{x} {y} H{x+w} V{y+h} H{x} Z', color, 1)


def grid(s, x, y, rows, cols, color=INK, cell=12):
    s.append(f'<rect x="{x}" y="{y}" width="{cols*cell}" height="{rows*cell}" '
             f'fill="{FILLS[color]}" stroke="{color}" stroke-width="1.6"/>')
    for row in range(1, rows):
        d.path(s, f'M{x} {y+row*cell} H{x+cols*cell}', color, .7)
    for col in range(1, cols):
        d.path(s, f'M{x+col*cell} {y} V{y+rows*cell}', color, .7)


def down(s, x, y, end, color=INK):
    d.path(s, f'M{x} {y} V{end} M{x-4} {end-7} L{x} {end} L{x+4} {end-7}', color, 1.4)


def mlp():
    s = d.start('The global MLP',
        'x, 8 by 4, times w_up, 4 by 6, followed by GeLU yields h, 8 by 6. '
        'h times w_down, 6 by 4, yields y, 8 by 4.', 205)
    for x, rows, cols, label in [
        (20,8,4,'x'), (111,4,6,'w_up'),
        (272,8,6,'h'), (395,6,4,'w_down'), (546,8,4,'y'),
    ]:
        y = 53+(8-rows)*7
        d.text(s, x+cols*7, 30, label, size=22)
        grid(s, x, y, rows, cols, INK, 14)
        d.text(s, x+cols*7, 193, f'{rows} × {cols}', size=19)
    for x in (94,377):
        d.text(s, x, 117, '×', size=25)
    d.arrow(s,205,108,260,108,INK,1.4)
    d.text(s,232,89,'GeLU',size=17)
    d.arrow(s,464,108,531,108,INK,1.4)
    d.save(s,'scaling3-mlp.png')


def pieces(kind):
    """(rows, cols, grid row, grid column, label, color) for each array."""
    batch_parts = 2 if kind == 'tp' else 4
    batch_rows = 8 // batch_parts
    xs = [(batch_rows,4,i,0,'x'+str(i),BATCH_COLORS[i])
          for i in range(batch_parts)]
    if kind == 'dp':
        us = [(4,6,0,0,'U',INK)]
        ds = [(6,4,0,0,'D',INK)]
    elif kind == 'fsdp':
        us = [(2,6,i,0,'U'+str(i),WEIGHT_COLORS[i]) for i in range(2)]
        ds = [(6,2,0,i,'D'+str(i),WEIGHT_COLORS[i]) for i in range(2)]
    else:
        us = [(4,3,0,i,'U'+str(i),WEIGHT_COLORS[i]) for i in range(2)]
        ds = [(3,4,i,0,'D'+str(i),WEIGHT_COLORS[i]) for i in range(2)]
    return xs, us, ds


def source(s, cx, cy, ps, label, shape):
    """Pull adjacent slices slightly apart so the cut direction is visible."""
    rr,cc=ps[0][:2]
    width=(max(p[3] for p in ps)+1)*(cc*12+6)-6
    height=(max(p[2] for p in ps)+1)*(rr*12+6)-6
    x=cx-width/2; y=cy-height/2
    d.text(s,cx,91,label,size=22)
    for rows,cols,r,c,name,color in ps:
        px=x+c*(cc*12+6); py=y+r*(rr*12+6)
        grid(s,px,py,rows,cols,color)
        # Labels sit beside row slices and below column slices.
        if max(p[3] for p in ps):
            d.text(s,px+cols*6,py+rows*12+18,name,color,16)
        else:
            d.text(s,px-9,py+rows*6+5,name,color,16,anchor='end')
    d.text(s,cx,241,shape,size=18)


def chip(s,x,y,device_id,ps):
    box(s,x,y,232,151,INK)
    for delta in (25,60,95,130,165,200):
        d.path(s,f'M{x+delta} {y-4} V{y} M{x+delta} {y+151} V{y+155}',FAINT,.9)
    d.text(s,x+12,y+18,str(device_id),FAINT,15)
    for cx,p in zip((x+44,x+122,x+198),ps):
        rows,cols,_,_,name,color=p
        d.text(s,cx,y+29,name,color,20)
        grid(s,cx-cols*6,y+78-rows*6,rows,cols,color)
        d.text(s,cx,y+139,f'{rows} × {cols}',color,17)


def placement(kind,title,mesh):
    xs,us,ds=pieces(kind)
    s=d.start(title,
        f'The same 8 by 4 x, 4 by 6 w_up, and 6 by 4 w_down on a {mesh} '
        'mesh in replica, fsdp, tensor order. Colored source pieces are pulled '
        'apart and recur in the device chips with their local shapes.',660)
    d.text(s,93,33,title,size=24,anchor='start')
    for cx,name,value in zip((365,465,565),('replica','fsdp','tensor'),mesh):
        d.text(s,cx,20,name,size=16)
        d.text(s,cx,49,str(value),BLUE if value>1 else FAINT,27)
    for cx,ps,label,shape in [(130,xs,'x','8 × 4'),
                             (320,us,'w_up','4 × 6'),
                             (510,ds,'w_down','6 × 4')]:
        source(s,cx,158,ps,label,shape)
    # A branching placement arrow connects the global arrays to both hosts.
    d.path(s,'M320 255 V266 M160 281 V266 H480 V281 M155 274 L160 281 L165 274 M475 274 L480 281 L485 274',INK,1.4)
    for host in range(2):
        hx=38+host*298
        box(s,hx,304,260,340)
        d.text(s,hx+130,298,f'Host {host}',size=20)
        for local in range(2):
            dev=host*2+local
            xp=xs[host if kind=='tp' else dev]
            wp=0 if kind=='dp' else local
            chip(s,hx+14,317+local*169,dev,(xp,us[wp],ds[wp]))
    d.save(s,f'scaling3-{kind}-placement.png')


if __name__=='__main__':
    d.OUT=Path(__file__).resolve().parents[1]/'assets/images/scaling'
    mlp()
    placement('dp','DP',(4,1,1))
    placement('fsdp','DP + FSDP',(2,2,1))
    placement('tp','DP + TP',(2,1,2))
