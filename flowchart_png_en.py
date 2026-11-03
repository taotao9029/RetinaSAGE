from PIL import Image, ImageDraw, ImageFont
import os, math

W, H = 2400, 1500
img = Image.new('RGB', (W, H), '#F8FAFC')
d = ImageDraw.Draw(img)
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
title = ImageFont.truetype(font_path, 46)
section = ImageFont.truetype(font_path, 30)

def box(x, y, w, h, text, fill, stroke='#64748B', fs=25):
    d.rounded_rectangle((x,y,x+w,y+h), radius=18, fill=fill, outline=stroke, width=3)
    f = ImageFont.truetype(font_path, fs)
    lines = text.split('\n')
    heights = [d.textbbox((0,0), line, font=f)[3] for line in lines]
    yy = y + (h-(sum(heights)+8*(len(lines)-1)))//2
    for line, hh in zip(lines, heights):
        bb = d.textbbox((0,0), line, font=f)
        d.text((x+(w-(bb[2]-bb[0]))//2, yy), line, fill='#0F172A', font=f)
        yy += hh + 8

def arrow(x1,y1,x2,y2,color='#475569',width=5):
    d.line((x1,y1,x2,y2), fill=color, width=width)
    a=math.atan2(y2-y1,x2-x1); L=18; ang=.45
    p1=(x2-L*math.cos(a-ang),y2-L*math.sin(a-ang)); p2=(x2-L*math.cos(a+ang),y2-L*math.sin(a+ang))
    d.polygon([(x2,y2),p1,p2], fill=color)

d.text((W//2-520,35),'OCTA Video Predictive State Modeling',fill='#0F172A',font=title)
d.text((W//2-500,95),'Raw Video -> Feature Representation -> Spatial/Temporal Modeling -> Prediction',fill='#475569',font=section)

box(70,220,260,125,'Raw OCTA Video','#DBEAFE','#2563EB')
box(390,220,300,125,'Preprocessing\nSampling + ECC Registration + Masks','#DBEAFE','#2563EB')
box(750,220,320,125,'Frozen DINOv3 ViT-S/16\n256x256 Frame Features','#DCFCE7','#16A34A')
arrow(330,282,390,282); arrow(690,282,750,282)
box(1140,150,320,125,'Global Features','#F1F5F9','#64748B')
box(1140,330,320,125,'Patch Features\n16x16 Patch Grid','#DCFCE7','#16A34A')
arrow(1070,260,1140,210); arrow(1070,305,1140,390)
box(1530,330,320,125,'4x4 Fixed Regions\n16 Local Regions','#DCFCE7','#16A34A')
arrow(1460,392,1530,392)
box(1910,330,390,125,'GATv2 Spatial Modeling\nLocal Region Relations','#FEF3C7','#D97706')
arrow(1850,392,1910,392)
box(1910,520,390,125,'Causal Temporal Transformer\nTime Encoding + Causal Mask','#FEF3C7','#D97706')
arrow(2105,455,2105,520)
box(1910,710,390,125,'128-D Video State\nReusable Representation','#F3E8FF','#9333EA')
arrow(2105,645,2105,710)
box(1540,720,280,115,'Video Classification\nDisease Logit','#F3E8FF','#9333EA')
arrow(1910,772,1820,772)
box(1540,900,280,115,'Future Local Feature\nPrediction: Delta E(t,k,h)','#FDE68A','#D97706',22)
arrow(2105,835,1820,940)
box(1220,730,235,72,'P1: Classification Only','#EDE9FE','#7C3AED',22)
box(1220,885,235,72,'P2: + Future Prediction','#EDE9FE','#7C3AED',22)
arrow(1540,777,1455,766); arrow(1540,957,1455,921)

box(740,590,330,110,'Fixed Reference Frame\nMulti-scale Frangi Response','#FCE7F3','#DB2777',22)
box(1120,590,300,110,'Structure-weighted Pooling\nReliable Region Mask','#FCE7F3','#DB2777',22)
arrow(905,345,905,590); arrow(1070,645,1120,645); arrow(1420,645,1760,645)
box(1220,1030,235,72,'P3: + Structure Prior','#FCE7F3','#DB2777',22)
arrow(1337,700,1337,1030)

d.text((80,1160),'Baseline Experiments',fill='#334155',font=section)
baseline=[('B1','Single-frame DINOv3'),('B2','Pixel temporal mean'),('B3','Feature temporal mean'),('B4','Deep Sets: unordered frames'),('B5','Global features + GRU'),('B6','Local features + Transformer\nwithout GAT'),('B8','RETFound'),('B9','V-JEPA 2.1')]
for i,(tag,txt) in enumerate(baseline):
    x=80+(i%4)*560; y=1220+(i//4)*120
    box(x,y,500,88,f'{tag}  {txt}','#F1F5F9','#94A3B8',22)

img.save('octa_video_pipeline_en.png', dpi=(200,200))
print('octa_video_pipeline_en.png')
