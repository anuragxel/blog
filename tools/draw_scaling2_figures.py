"""Draw tensor and pipeline parallelism figures using the blog's SVG/Cairo style.

Run: /usr/bin/python3 tools/draw_scaling2_figures.py
"""
from pathlib import Path
import draw_transformer_figures as draw
from draw_fsdp_figure import rect, down

INK, BLUE, GREEN, FAINT = draw.INK, draw.BLUE, draw.GREEN, draw.FAINT


def tile(s,x,y,w,h,color,label,filled=True):
    rect(s,x,y,w,h,color,fill=('#eeeeff' if color==BLUE else '#eaf5ee') if filled else 'white')
    draw.text(s,x+w/2,y+h/2+9,label,color,24)


def array_grid(s, x, y, rows, cols, color, cell=14):
    """Each visible cell is one scalar, with rows and columns at equal scale."""
    fill = {BLUE: '#eeeeff', GREEN: '#eaf5ee'}.get(color, '#f0f0f2')
    s.append(f'<rect x="{x}" y="{y}" width="{cols*cell}" height="{rows*cell}" fill="{fill}" stroke="{color}" stroke-width="1.6"/>')
    for row in range(1, rows):
        draw.path(s, f'M{x} {y+row*cell} H{x+cols*cell}', color, .7)
    for col in range(1, cols):
        draw.path(s, f'M{x+col*cell} {y} V{y+rows*cell}', color, .7)


def tensor():
    s = draw.start('Tensor parallel MLP, forward pass',
        'A is a 4 by 6 matrix split into two 4 by 3 column slices. B is a 6 by 4 matrix split into matching 3 by 4 row slices. Each device multiplies the same 3 by 4 X by its A slice, applies GeLU to the 3 by 3 result, and multiplies by its B slice. All-reduce sums the two 3 by 4 partial outputs.', 565)
    draw.text(s,320,35,'Tensor parallelism',size=28)
    draw.text(s,192,79,'A  (4 × 6)',size=23)
    draw.text(s,448,79,'B  (6 × 4)',size=23)
    for i,color in enumerate((BLUE,GREEN)):
        array_grid(s,150+i*42,109,4,3,color)
        draw.text(s,171+i*42,190,('A₀','A₁')[i],color,22)
        array_grid(s,420,95+i*42,3,4,color)
        draw.text(s,498,125+i*42,('B₀','B₁')[i],color,22)
    for i,(y,color) in enumerate(((252,BLUE),(387,GREEN))):
        sub=('₀','₁')[i]
        draw.text(s,24,y-28,f'Device {i}',color,21,anchor='start')
        arrays = ((24,3,4,INK,'X'), (118,4,3,color,'A'+sub),
                  (257,3,3,color,'H'+sub), (351,3,4,color,'B'+sub),
                  (469,3,4,color,'Y'+sub))
        for x,rows,cols,ink,label in arrays:
            top=y+(56-rows*14)/2
            array_grid(s,x,top,rows,cols,ink)
            draw.text(s,x+cols*7,y-6,label,ink,23)
            draw.text(s,x+cols*7,y+78,f'{rows} × {cols}',ink,17)
        draw.text(s,99,y+37,'×',size=25)
        draw.arrow(s,174,y+28,241,y+28,color,1.6)
        draw.text(s,208,y+11,'GeLU',size=18)
        draw.text(s,325,y+37,'×',size=25)
        draw.arrow(s,421,y+28,455,y+28,color,1.6)
    draw.path(s,'M539 280 H582 V335 M539 415 H582 V360',INK,1.6)
    draw.text(s,582,359,'+',size=36)
    draw.path(s,'M582 369 V510 H350 M358 505 L350 510 L358 515',INK,1.6)
    array_grid(s,280,489,3,4,INK)
    draw.text(s,308,481,'Y',size=25)
    draw.text(s,308,555,'3 × 4',size=18)
    draw.text(s,451,542,'all-reduce',size=22)
    draw.save(s,'scaling1-tensor-parallel.png')


def schedule(stages, microbatches):
    """GPipe flush with FIFO microbatch order in both passes."""
    length = 2*(microbatches+stages-1)
    grid = [[None]*length for _ in range(stages)]
    forward_end = microbatches+stages-1
    for stage in range(stages):
        for micro in range(microbatches):
            ft = stage+micro
            bt = forward_end+(stages-1-stage)+micro
            assert grid[stage][ft] is None and grid[stage][bt] is None
            grid[stage][ft] = ('F', micro+1)
            grid[stage][bt] = ('B', micro+1)
            assert bt > ft
            if stage:
                assert ft > (stage-1)+micro
            if stage < stages-1:
                next_bt = forward_end+(stages-2-stage)+micro
                assert bt > next_bt
    assert all(sum(cell is not None for cell in row)==2*microbatches for row in grid)
    return grid


def schedule_1f1b(stages, microbatches):
    """Synchronous 1F1B with warmup, alternating work, and drain."""
    queues = []
    for stage in range(stages):
        warmup = min(stages-stage-1, microbatches)
        queue = [('F', m+1) for m in range(warmup)]
        for m in range(microbatches-warmup):
            queue.extend([('F', warmup+m+1), ('B', m+1)])
        queue.extend(('B', m+1) for m in range(microbatches-warmup, microbatches))
        queues.append(queue)
    grid = [[] for _ in range(stages)]
    done = set()
    while any(queues):
        ready = []
        for stage, queue in enumerate(queues):
            cell = None
            if queue:
                kind, micro = queue[0]
                dependencies = []
                if kind == 'F' and stage:
                    dependencies.append((stage-1, 'F', micro))
                if kind == 'B':
                    dependencies.append((stage, 'F', micro))
                    if stage < stages-1:
                        dependencies.append((stage+1, 'B', micro))
                if all(dep in done for dep in dependencies):
                    cell = queue.pop(0)
                    ready.append((stage, *cell))
            grid[stage].append(cell)
        assert ready, 'Pipeline dependency deadlock'
        done.update(ready)
    return grid


def pipeline():
    s = draw.start('GPipe and 1F1B on the same four microbatches',
        'Four microbatches on three devices with equal forward and backward costs and no communication overhead. GPipe completes all forwards before backwards. 1F1B alternates forward and backward work after warmup. Both take twelve time slots, but 1F1B releases saved activations earlier.', 470)
    s.append('<defs>')
    for name,color in (('forward',BLUE),('backward',GREEN)):
        s.append(f'<pattern id="{name}" width="7" height="7" patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="white"/><path d="M-1 1 L1 -1 M0 7 L7 0 M6 8 L8 6" stroke="{color}" stroke-width="0.8"/></pattern>')
    s.append('</defs>')
    draw.text(s,320,25,'4 microbatches',size=23)
    for x,label,color in ((160,'forward',BLUE),(320,'backward',GREEN),(490,'idle','#ededf0')):
        rect(s,x-25,45,17,17,color,fill=color if label=='idle' else f'url(#{label})')
        draw.text(s,x,60,label,size=17,anchor='start')
    for y,label,grid in ((110,'GPipe',schedule(3,4)),(285,'1F1B (Practical Schedule Refinement)',schedule_1f1b(3,4))):
        draw.text(s,110,y,label,size=20 if label.startswith('1F1B') else 23,anchor='start')
        for stage,row in enumerate(grid):
            ry = y+20+stage*33
            draw.text(s,94,ry+20,f'Device {stage}',size=16,anchor='end')
            draw.text(s,532,ry+20,f'layers {stage*4+1}–{stage*4+4}',size=14,anchor='start')
            for t,cell in enumerate(row):
                x = 110+t*34
                color = '#ededf0' if cell is None else BLUE if cell[0]=='F' else GREEN
                fill = color if cell is None else 'url(#forward)' if cell[0]=='F' else 'url(#backward)'
                rect(s,x,ry,30,27,color,fill=fill)
                if cell:
                    s.append(f'<rect x="{x+8}" y="{ry+4}" width="14" height="19" rx="3" fill="white"/>')
                    draw.text(s,x+15,ry+20,str(cell[1]),color,size=17)
    draw.arrow(s,110,435,514,435,FAINT,1.4)
    draw.text(s,312,460,'time',size=16)
    draw.save(s,'scaling1-pipeline.png')


if __name__ == '__main__':
    draw.OUT=Path(__file__).resolve().parents[1]/'assets/images/scaling'
    tensor()
    pipeline()
