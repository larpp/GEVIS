"""사용자가 지정한 아키텍처 그림의 구성을 독립 SVG로 재현한다."""
from pathlib import Path
from html import escape


def draw_architecture(destination):
    parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="1783" height="589" viewBox="0 0 1783 589">
<defs>
 <linearGradient id="blue" x2="0" y2="1"><stop stop-color="#f3faff"/><stop offset="1" stop-color="#e4f2ff"/></linearGradient>
 <linearGradient id="purple" x2="0" y2="1"><stop stop-color="#faf6ff"/><stop offset="1" stop-color="#f0e7ff"/></linearGradient>
 <linearGradient id="pink" x2="0" y2="1"><stop stop-color="#fff8fd"/><stop offset="1" stop-color="#ffeaf7"/></linearGradient>
 <linearGradient id="green" x2="0" y2="1"><stop stop-color="#f5fff8"/><stop offset="1" stop-color="#e8f8ed"/></linearGradient>
 <linearGradient id="strongblue" x2="0" y2="1"><stop stop-color="#1da0ff"/><stop offset="1" stop-color="#0065ef"/></linearGradient>
 <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="#087bff"/></marker>
 <marker id="violetarrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7Z" fill="#9250fb"/></marker>
</defs>
<rect width="1783" height="589" fill="white"/>
<style>text{font-family:Arial,'DejaVu Sans',sans-serif;fill:#071650} .bold{font-weight:700} .line{fill:none;stroke:#087bff;stroke-width:3;marker-end:url(#arrow)} .vline{fill:none;stroke:#9250fb;stroke-width:3;marker-end:url(#violetarrow)}</style>''']
    def rect(x,y,w,h,fill,stroke='#78b4ff',radius=18,dash=None):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.7"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def text(x,y,value,size=23,bold=False,anchor='start',color=None):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"'+(' class="bold"' if bold else '')+(f' style="fill:{color}"' if color else '')+'>'+escape(value)+'</text>')
    def arrow(x1,y1,x2,y2):parts.append(f'<path class="line" d="M{x1} {y1} L{x2} {y2}"/>')
    text(54,44,'GEVIS Architecture (Updated)',42,True,color='#1b3472')
    rect(13,85,342,449,'url(#blue)','#1685ff',24)
    text(34,131,'⚙',38,True,color='#1596ef')
    text(100,124,'ALD Preprocessing',26,True);text(100,151,'(Separated)',25,True)
    for y,title,sub in [(182,'ALD Data','(e.g., PEALD)'),(302,'Preprocess','(Sensor · Norm.)'),(425,'Normal Range','(MIN / MAX)')]:
        rect(41,y,289,89,'#f8f8ff','#c6d4ed')
        text(151,y+37,title,23,True);text(151,y+65,sub,22)
    # Vector icons: database, controls and range bars.
    parts.append('<g fill="none" stroke="#7851bc" stroke-width="3"><ellipse cx="93" cy="211" rx="20" ry="10"/><path d="M73 211V241C73 254 113 254 113 241V211 M73 226C73 239 113 239 113 226"/></g>')
    parts.append('<g stroke="#7851bc" stroke-width="4"><path d="M77 330H120 M77 347H120 M77 364H120"/></g><g fill="#f8f8ff" stroke="#7851bc" stroke-width="4"><circle cx="90" cy="330" r="5"/><circle cx="106" cy="347" r="5"/><circle cx="96" cy="364" r="5"/></g>')
    for i,h in enumerate([23,44,34]):rect(73+i*16,490-h,10,h,'#835ac7','#835ac7',2)
    arrow(190,273,190,297);arrow(190,393,190,420);arrow(357,322,387,322)
    rect(389,73,715,481,'#fcfdff','#258cff',27,'10 8')
    parts.append('<g fill="#158cf4"><rect x="424" y="109" width="34" height="28" rx="5"/><path d="M431 110V102A10 10 0 0 1 451 102V110H446V102A5 5 0 0 0 436 102V110Z"/></g>')
    text(487,118,'GEVIS (Feature Extractor, Fixed)',32,True)
    text(487,148,'Trained on PECVD (6 Step Encoders)',24)
    rect(404,172,227,347,'url(#blue)','#58a8ff')
    text(517,205,'Time-series',25,True,'middle');text(517,234,'Encoder',25,True,'middle')
    text(517,263,'(PECVD, 6 Steps)',22,False,'middle')
    for i,name in enumerate(['Gap-up','PRE-HEAT','STAB1','DEP. (Used)','PURGE','PUMPING']):
        y=278+i*37;active=i==3
        rect(421,y,191,31,'#086dff' if active else '#f5faff','#74b8ff',9)
        text(516,y+22,name,19,not active,'middle',color='white' if active else None)
    arrow(633,322,661,322)
    rect(663,195,225,203,'url(#purple)','#ab7cff')
    text(775,229,'Fusion',28,True,'middle');text(775,257,'(Cross-Attention)',21,False,'middle')
    for y,label in [(277,'Q'),(315,'K'),(353,'V')]:
        rect(691,y,50,32,'#ffe3f8','#f3a4e5',8);text(716,y+24,label,24,True,'middle')
    parts.append('<g class="vline"><path d="M746 291C784 291 783 319 823 319"/><path d="M746 331H823"/><path d="M746 369C784 369 783 342 823 342"/></g>')
    parts.append('<circle cx="847" cy="330" r="21" fill="#fcf7ff" stroke="#a36cff" stroke-width="3"/><path d="M836 330H858 M847 319V341" stroke="#a36cff" stroke-width="3"/>')
    rect(656,446,315,83,'url(#purple)','#ba8bff')
    parts.append('<path d="M677 463H702L714 476V510H677Z M702 463V477H714 M685 486H704 M685 497H704" stroke="#087bff" fill="none" stroke-width="3"/>')
    text(744,480,'Question Encoder',25,True);text(744,510,'(E5-small-v2, Frozen)',23)
    parts.append('<path class="vline" d="M828 444V400"/>')
    arrow(890,322,920,322)
    rect(922,217,169,183,'url(#blue)','#58a8ff')
    text(1007,253,'GEVIS Feature',22,True,'middle');text(1007,281,'(768-d)',23,False,'middle')
    for i,color in enumerate(['#91b4ff','#b99bff','#ffe27d']):rect(941+i*26,318,26,35,color,color,4)
    text(1064,340,'···',28,True,'middle');arrow(1095,322,1127,322)
    rect(1129,182,252,304,'url(#blue)','#58a8ff',23)
    text(1255,215,'Feature Assembly',26,True,'middle');text(1255,243,'(1,546-d)',24,True,'middle')
    for y,label,size,fill in [(264,'GEVIS (MIN)','768-d','url(#purple)'),(317,'GEVIS (MAX)','768-d','url(#purple)'),(370,'Boundary Stats','8-d','url(#blue)'),(421,'Time Info','2-d','url(#blue)')]:
        rect(1147,y,216,43,fill,'#bba0f9' if 'GEVIS' in label else '#9dd7f4',11)
        text(1159,y+28,label,19,True);text(1352,y+28,size,20,False,'end')
    arrow(1383,322,1407,322)
    rect(1399,85,382,444,'url(#purple)','#7731ef',24)
    parts.append('<g fill="white" stroke="#6a28bc" stroke-width="3"><circle cx="1440" cy="111" r="6"/><circle cx="1430" cy="141" r="6"/><circle cx="1452" cy="141" r="6"/><path d="M1440 117L1430 135 M1440 117L1452 135"/></g>')
    text(1470,118,'ALD Decision Heads',27,True);text(1470,147,'(Trained on ALD)',24)
    for y,title,subtitle,fill,stroke,color,icon in [
        (190,'BOOLEAN Head','(2 classes)','url(#blue)','#80bfff','#2b9cf7','check'),
        (298,'CHOICE Head','(6 classes)','url(#pink)','#f4a1dd','#ef5ab6','list'),
        (406,'ORDINAL Head','(5 levels)','url(#green)','#8dd4aa','#19bf6b','bars')]:
        rect(1425,y,334,89,fill,stroke)
        parts.append(f'<circle cx="1472" cy="{y+43}" r="29" fill="{color}"/>')
        text(1527,y+37,title,28,True);text(1527,y+67,subtitle,23)
        if icon=='check':parts.append(f'<path d="M1457 {y+42}L1467 {y+52}L1487 {y+31}" fill="none" stroke="white" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>')
        elif icon=='list':
            for offset in [31,43,55]:parts.append(f'<circle cx="1459" cy="{y+offset}" r="2" fill="white"/><path d="M1467 {y+offset}H1486" stroke="white" stroke-width="3"/>')
        else:
            for i,h in enumerate([10,19,29]):rect(1456+i*11,y+58-h,7,h,'white','white',1)
    parts.append('</svg>')
    Path(destination).write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':
    draw_architecture(Path(__file__).resolve().parents[1]/'docs/architecture/gevis.svg')
