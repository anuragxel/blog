"""Pictorial SSL diagrams, rendered to 2x PNG with shared Cairo helpers.

Run /usr/bin/python3 tools/draw_scaling1_figures.py.
"""
from math import atan2, cos, sin
from pathlib import Path
import draw_transformer_figures as d

BLUE, GREEN, INK, FAINT = d.BLUE, d.GREEN, d.INK, d.FAINT
OUT = Path(__file__).resolve().parents[1] / 'assets/images/scaling'


def save(s, name):
    old = d.OUT
    try:
        d.OUT = OUT
        d.save(s, name)
    finally:
        d.OUT = old


def arrow(s, x, y, xx, yy, color=INK, dash=False):
    d.path(s, f'M{x} {y} L{xx} {yy}', color, 1.7, dash)
    a = atan2(yy-y, xx-x)
    d.path(s, f'M{xx-7*cos(a-.5)} {yy-7*sin(a-.5)} L{xx} {yy} L{xx-7*cos(a+.5)} {yy-7*sin(a+.5)}', color, 1.7)


def scene(s, x, y, w, h=None, crop='0 0 100 80', kind=0):
    h = h or w*.8
    s.append(f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{crop}" preserveAspectRatio="none" overflow="hidden">')
    s.append('<rect width="100" height="80" fill="#f4f7ff"/>')
    if kind == 0:
        s.append(f'<circle cx="77" cy="17" r="8" fill="#dbe8da" stroke="{GREEN}" stroke-width="1.4"/>')
        s.append(f'<path d="M-5 65 L30 16 L70 65 L82 43 L108 76 H-5 Z" fill="#e0e2f7" stroke="{BLUE}" stroke-width="1.7"/>')
        s.append(f'<path d="M19 32 L30 16 L43 33 L32 29 L27 34 Z" fill="white" stroke="{BLUE}" stroke-width="1.2"/>')
        s.append(f'<path d="M0 72 Q45 53 100 71 V80 H0 Z" fill="#e3efdf" stroke="{GREEN}" stroke-width="1.4"/>')
        s.append(f'<path d="M75 67 V43 M61 54 L75 32 L89 54 Z M63 44 L75 25 L87 44 Z" fill="#d2e6cc" stroke="{GREEN}" stroke-width="1.7"/>')
    elif kind == 1:
        s.append(f'<path d="M0 53 Q30 46 50 54 T100 54 V80 H0 Z" fill="#e0e2f7" stroke="{BLUE}" stroke-width="1.5"/>')
        s.append(f'<path d="M24 51 H77 L65 65 H37 Z M50 48 V13 L74 44 H50 M45 20 L27 44 H45 Z" fill="white" stroke="{GREEN}" stroke-width="2"/>')
    else:
        s.append(f'<path d="M0 72 H100 M20 69 V36 H76 V69 M13 36 L47 12 L84 36 Z M43 69 V48 H57 V69" fill="#e3efdf" stroke="{GREEN}" stroke-width="2"/>')
        s.append(f'<path d="M28 44 H36 V52 H28 Z M62 44 H70 V52 H62 Z" fill="white" stroke="{BLUE}" stroke-width="1.5"/>')
    s.append('</svg>')
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="none" stroke="{BLUE}" stroke-width="1.5"/>')


def encoder(s, x, y, w=44, h=64, color=BLUE):
    for off in (8, 4, 0):
        s.append(f'<rect x="{x+off}" y="{y-off}" width="{w}" height="{h}" rx="4" fill="white" stroke="{color}" stroke-width="1.5"/>')


def point(s, x, y, color=BLUE, radius=5):
    s.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="white" stroke="{color}" stroke-width="2"/>')


def simclr():
    s=d.start('SimCLR brings two crops together', 'Two distinct crops of a mountain and tree image pass through one encoder and land near each other in embedding space. A boat and a house occupy distant locations.', 310)
    scene(s, 20, 112, 112)
    for x,y,crop in [(187,49,'0 0 66 64'),(187,197,'42 18 58 60')]:
        scene(s,x,y,73,66,crop)
        arrow(s,137,150,x-9,y+33)
        arrow(s,267,y+33,307,151)
    encoder(s,313,122,39,58)
    d.text(s,337,282,'shared encoder',BLUE,17)
    arrow(s,364,151,431,151)
    s.append(f'<ellipse cx="525" cy="152" rx="92" ry="103" fill="none" stroke="{FAINT}" stroke-dasharray="4 6"/>')
    point(s,475,144,BLUE)
    point(s,487,155,GREEN)
    d.path(s,'M468 135 Q490 128 497 153 Q500 169 481 169',GREEN,1.3)
    scene(s,543,45,45,36,kind=1)
    point(s,565,96)
    scene(s,546,215,45,36,kind=2)
    point(s,563,201)
    d.text(s,480,190,'same image',GREEN,17)
    save(s,'scaling0-simclr.png')


def masked(s,x,y):
    scene(s,x,y,88,80)
    visible={0,6,9,15}
    for i in range(16):
        xx,yy=x+i%4*22,y+i//4*20
        if i not in visible:
            s.append(f'<rect x="{xx}" y="{yy}" width="22" height="20" fill="#f1f1f3"/>')
        s.append(f'<rect x="{xx}" y="{yy}" width="22" height="20" fill="none" stroke="white" stroke-width="1.5"/>')


def masking():
    s=d.start('MAE drops hidden patches before the encoder', 'MAE sends visible image patches into its encoder and adds blank mask tokens at the decoder. SimMIM carries visible and masked positions through its encoder. Both reconstruct the scene.', 335)
    for y,label in [(64,'MAE'),(222,'SimMIM')]:
        d.text(s,62,y-23,label,BLUE,23)
        masked(s,18,y)
        arrow(s,111,y+40,143,y+40)
        if label=='MAE':
            for j,i in enumerate([0,6,9,15]):
                scene(s,151+j*24,y+28,20,22,f'{i%4*25} {i//4*20} 25 20')
        else:
            for i in range(16):
                xx,yy=152+i%4*23,y+3+i//4*20
                if i in {0,6,9,15}:
                    scene(s,xx,yy,20,17,f'{i%4*25} {i//4*20} 25 20')
                else:
                    s.append(f'<rect x="{xx}" y="{yy}" width="20" height="17" fill="#f1f1f3" stroke="{FAINT}" stroke-width="1"/>')
        arrow(s,252,y+40,282,y+40)
        encoder(s,290,y+9,54,63)
        arrow(s,354,y+40,401,y+40)
        if label=='MAE':
            encoder(s,411,y+19,36,44,GREEN)
            for j in range(3):
                s.append(f'<rect x="{366+j*13}" y="{y+83}" width="10" height="10" fill="#f1f1f3" stroke="{FAINT}"/>')
            arrow(s,404,y+86,426,y+69,FAINT)
        else:
            s.append(f'<rect x="419" y="{y+17}" width="12" height="47" rx="3" fill="white" stroke="{GREEN}" stroke-width="2"/>')
        arrow(s,461,y+40,503,y+40,GREEN)
        scene(s,514,y,100,80)
    d.text(s,319,24,'encoder',BLUE,18)
    d.text(s,430,24,'decoder',GREEN,18)
    save(s,'scaling0-mae-simmim.png')


def histogram(s,x,y):
    for i,h in enumerate([12,24,58,16,9]):
        s.append(f'<rect x="{x+i*15}" y="{y-h}" width="10" height="{h}" rx="2" fill="{GREEN if i==2 else "#dee0f4"}" stroke="{GREEN if i==2 else BLUE}" stroke-width="1"/>')
    d.path(s,f'M{x-4} {y+4} H{x+77}',FAINT,1)


def dino():
    s=d.start('DINO matches a teacher seeing more of the image', 'A teacher receives a large crop while a student receives a smaller crop. Their output distributions match. A dashed EMA arrow points from student weights to teacher weights.', 325)
    scene(s,35,36,144,110)
    scene(s,81,226,70,64,'42 18 58 60')
    for y,label in [(95,'teacher'),(251,'student')]:
        arrow(s,190 if label=='teacher' else 163,y,273,y)
        encoder(s,282,y-32,59,65)
        d.text(s,315,y-47,label,BLUE,21)
        arrow(s,356,y,443,y)
        histogram(s,466,y+30)
    d.path(s,'M342 221 Q413 159 347 129',FAINT,1.5,True)
    arrow(s,347,129,342,126,FAINT,True)
    d.text(s,392,174,'EMA',INK,17)
    d.path(s,'M556 96 H579 V251 H556',GREEN,1.6)
    d.text(s,591,179,'≈',GREEN,30)
    save(s,'scaling0-dino.png')


def geometry():
    s=d.start('SimDINO spreads matching pairs across directions', 'Illustrative normalized embedding geometry. Three different images all map to one point on the left unit circle. On the right, their matching view pairs occupy different directions on a unit circle.', 305)
    for offset,label in [(0,'collapse'),(320,'spread')]:
        d.text(s,offset+188,37,label,BLUE if not offset else GREEN,23)
        cx,cy=offset+209,167
        s.append(f'<circle cx="{cx}" cy="{cy}" r="86" fill="none" stroke="{FAINT}" stroke-width="1.5"/>')
        s.append(f'<ellipse cx="{cx}" cy="{cy}" rx="86" ry="25" fill="none" stroke="#dedee2" stroke-width="1.1"/>')
        for i,ang in enumerate([-.7,1.4,3.3]):
            y=70+i*76
            scene(s,offset+15,y,52,42,kind=i)
            a=-2.6 if not offset else ang
            px,py=cx+86*cos(a),cy+86*sin(a)
            arrow(s,offset+73,y+21,px-10,py,FAINT)
            point(s,px,py,BLUE,7)
            s.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{GREEN}"/>')
    save(s,'scaling0-simdino.png')


if __name__=='__main__':
    simclr()
    masking()
    dino()
    geometry()
