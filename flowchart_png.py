from PIL import Image, ImageDraw, ImageFont
import os, math

W, H = 2400, 1500
img = Image.new('RGB', (W, H), '#F8FAFC')
d = ImageDraw.Draw(img)
font_paths = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
font_path = next((p for p in font_paths if os.path.exists(p)), font_paths[-1])
title = ImageFont.truetype(font_path, 46)
section = ImageFont.truetype(font_path, 30)

def box(x, y, w, h, text, fill, stroke='#64748B', fs=25):
    d.rounded_rectangle((x, y, x+w, y+h), radius=18, fill=fill, outline=stroke, width=3)
    f = ImageFont.truetype(font_path, fs)
    lines = text.split('\n')
    heights = [d.textbbox((0,0), line, font=f)[3] for line in lines]
    yy = y + (h-(sum(heights)+8*(len(lines)-1)))//2
    for line, hh in zip(lines, heights):
        bb = d.textbbox((0,0), line, font=f)
        d.text((x+(w-(bb[2]-bb[0]))//2, yy), line, fill='#0F172A', font=f)
        yy += hh + 8

def arrow(x1, y1, x2, y2, color='#475569', width=5):
    d.line((x1,y1,x2,y2), fill=color, width=width)
    a = math.atan2(y2-y1, x2-x1); L, ang = 18, 0.45
    p1=(x2-L*math.cos(a-ang), y2-L*math.sin(a-ang)); p2=(x2-L*math.cos(a+ang), y2-L*math.sin(a+ang))
    d.polygon([(x2,y2),p1,p2], fill=color)

d.text((W//2-520, 35), 'OCTA Video Predictive State Modeling', fill='#0F172A', font=title)
d.text((W//2-400, 95), '原始视频 → 特征表示 → 空间/时间建模 → 分类与未来预测', fill='#475569', font=section)

box(70,220,260,125,'原始眼底视频\nRaw OCTA Video','#DBEAFE','#2563EB')
box(390,220,300,125,'视频预处理\n采样 · ECC配准 · 掩码','#DBEAFE','#2563EB')
box(750,220,320,125,'冻结 DINOv3 ViT-S/16\n256×256 · 逐帧特征','#DCFCE7','#16A34A')
arrow(330,282,390,282); arrow(690,282,750,282)
box(1140,150,320,125,'全局特征\nGlobal Feature','#F1F5F9','#64748B')
box(1140,330,320,125,'Patch特征\n16×16 Patch Grid','#DCFCE7','#16A34A')
arrow(1070,260,1140,210); arrow(1070,305,1140,390)
box(1530,330,320,125,'4×4固定区域\n16个局部区域','#DCFCE7','#16A34A')
arrow(1460,392,1530,392)
box(1910,330,390,125,'GATv2空间建模\n局部区域关系','#FEF3C7','#D97706')
arrow(1850,392,1910,392)
box(1910,520,390,125,'因果时间Transformer\n时间编码 + 上三角Mask','#FEF3C7','#D97706')
arrow(2105,455,2105,520)
box(1910,710,390,125,'128维视频状态\nVideo State','#F3E8FF','#9333EA')
arrow(2105,645,2105,710)
box(1540,720,280,115,'视频级分类\nDisease Logit','#F3E8FF','#9333EA')
arrow(1910,772,1820,772)
box(1540,900,280,115,'未来局部特征预测\nΔE(t,k,h)','#FDE68A','#D97706')
arrow(2105,835,1820,940)
box(1220,730,235,72,'P1：仅分类','#EDE9FE','#7C3AED',22)
box(1220,885,235,72,'P2：+未来预测','#EDE9FE','#7C3AED',22)
arrow(1540,777,1455,766); arrow(1540,957,1455,921)

box(740,590,330,110,'固定参考帧\n多尺度 Frangi 响应','#FCE7F3','#DB2777')
box(1120,590,300,110,'结构加权区域池化\n可靠区域掩码','#FCE7F3','#DB2777')
arrow(905,345,905,590); arrow(1070,645,1120,645); arrow(1420,645,1760,645)
box(1220,1030,235,72,'P3：+结构先验','#FCE7F3','#DB2777',22)
arrow(1337,700,1337,1030)

d.text((80,1160), 'Baseline Experiments（基线与对照）', fill='#334155', font=section)
baseline = [('B1','单帧 DINOv3'),('B2','像素时间平均'),('B3','特征时间平均'),('B4','Deep Sets 无序多帧'),('B5','全局特征 + GRU'),('B6','局部特征 + Transformer\n去除GAT'),('B8','RETFound'),('B9','V-JEPA 2.1')]
for i,(tag,txt) in enumerate(baseline):
    x=80+(i%4)*560; y=1220+(i//4)*120
    box(x,y,500,88,f'{tag}  {txt}', '#F1F5F9', '#94A3B8', 22)

out='octa_video_pipeline.png'
img.save(out, dpi=(200,200))
print(out)
