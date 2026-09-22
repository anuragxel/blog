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


def tensor():
    s = draw.start('Tensor parallel MLP, forward pass',
        'A is split into blue and green columns, B into matching rows. Each device multiplies the same X by its A slice, applies GeLU, and multiplies by its B slice. The partial output matrices are summed by all-reduce.', 530)
    draw.text(s,320,35,'Tensor parallelism',size=28)
    draw.text(s,190,77,'A',size=25)
    draw.text(s,450,77,'B',size=25)
    for i,color in enumerate((BLUE,GREEN)):
        tile(s,132+i*60,92,56,76,color,('A₀','A₁')[i])
        tile(s,396,92+i*40,110,36,color,('B₀','B₁')[i])
    for i,(y,color) in enumerate(((247,BLUE),(372,GREEN))):
        sub=('₀','₁')[i]
        draw.text(s,29,y-29,f'Device {i}',color,21,anchor='start')
        draw.text(s,48,y+36,'X',size=29)
        draw.text(s,86,y+36,'×',size=25)
        tile(s,108,y,47,61,color,'A'+sub)
        draw.arrow(s,167,y+31,220,y+31,color,1.6)
        draw.text(s,194,y+11,'GeLU',size=18)
        tile(s,233,y+12,44,38,color,'H'+sub)
        draw.text(s,303,y+36,'×',size=25)
        tile(s,329,y+16,76,30,color,'B'+sub)
        draw.arrow(s,417,y+31,452,y+31,color,1.6)
        tile(s,466,y,80,61,color,'Y'+sub)
    draw.path(s,'M556 278 H582 V331 M556 403 H582 V351',INK,1.6)
    draw.text(s,582,350,'+',size=36)
    draw.path(s,'M582 361 V466 H346 M354 461 L346 466 L354 471',INK,1.6)
    tile(s,256,442,80,58,INK,'Y',filled=False)
    draw.text(s,441,494,'all-reduce',size=22)
    draw.save(s,'scaling1-tensor-parallel.png')


def schedule(stages, microbatches):
    """GPipe flush with reversed microbatch order in backward."""
    length = 2*(microbatches+stages-1)
    grid = [[None]*length for _ in range(stages)]
    forward_end = microbatches+stages-1
    for stage in range(stages):
        for micro in range(microbatches):
            ft = stage+micro
            bt = forward_end+(stages-1-stage)+(microbatches-1-micro)
            assert grid[stage][ft] is None and grid[stage][bt] is None
            grid[stage][ft] = ('F', micro+1)
            grid[stage][bt] = ('B', micro+1)
            assert bt > ft
            if stage:
                assert ft > (stage-1)+micro
            if stage < stages-1:
                next_bt = forward_end+(stages-2-stage)+(microbatches-1-micro)
                assert bt > next_bt
    assert all(sum(cell is not None for cell in row)==2*microbatches for row in grid)
    return grid


def pipeline():
    s = draw.start('GPipe pipeline schedules with one and four microbatches',
        'Three stages with equal forward and backward costs and no communication overhead. Forward operations are blue, backward operations green. Blank gray cells are idle. A single microbatch takes six slots with a two-thirds bubble. Four microbatches take twelve slots with a one-third bubble. Each backward starts after all forward operations finish.', 590)
    draw.text(s,320,34,'Pipeline parallelism: fill, work, drain',size=25)
    draw.text(s,320,63,'3 stages • equal F and B costs • no communication overhead',size=16)
    for y,m in ((112,1),(341,4)):
        grid=schedule(3,m)
        draw.text(s,24,y,f'{m} microbatch'+('' if m==1 else 'es'),size=22,anchor='start')
        draw.text(s,614,y,'idle bubble = '+('2/3' if m==1 else '1/3'),size=20,anchor='end')
        for stage,row in enumerate(grid):
            draw.text(s,80,y+51+stage*43,f'Stage {stage}',size=17,anchor='end')
            for t,cell in enumerate(row):
                x=96+t*42
                ry=y+25+stage*43
                color=FAINT if cell is None else BLUE if cell[0]=='F' else GREEN
                rect(s,x,ry,38,36,color,fill='#f4f4f5' if cell is None else 'white')
                if cell:
                    draw.text(s,x+19,ry+25,cell[0]+str(cell[1]),color,18)
        end=96+len(grid[0])*42-4
        draw.arrow(s,96,y+169,end,y+169,FAINT,1.4)
        draw.text(s,(96+end)/2,y+192,'time → '+str(len(grid[0]))+' slots',size=16)
        split=96+(m+2)*42-2
        draw.path(s,f'M{split} {y+18} V{y+151}',INK,1,dash=True)
        if m==1:
            draw.text(s,494,y+66,'F1 → forward',BLUE,18)
            draw.text(s,494,y+95,'B1 → backward',GREEN,18)
            draw.text(s,494,y+126,'gray → idle',size=18)
    draw.text(s,320,567,'Flush schedule: finish all forwards, then run backwards.',size=17)
    draw.save(s,'scaling1-pipeline.png')


if __name__ == '__main__':
    draw.OUT=Path(__file__).resolve().parents[1]/'assets/images/scaling'
    tensor()
    pipeline()
