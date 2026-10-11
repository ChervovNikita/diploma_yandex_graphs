from pathlib import Path
import re,html,json,sys
W=Path(__file__).resolve().parent
for name in sys.argv[1:]:
 s=(W/'primary'/f'{name}.raw').read_text()
 s=re.sub(r'<math\b[^>]*alttext="([^"]*)"[^>]*>.*?</math>',lambda m:' '+html.unescape(m.group(1))+' ',s,flags=re.S)
 blocks=[]
 for m in re.finditer(r'<(h[1-6]|p|table)\b([^>]*)>(.*?)</\1>',s,re.S):
  attrs=m.group(2);txt=re.sub('<[^>]+>',' ',m.group(3));txt=' '.join(html.unescape(txt).split())
  if not txt:continue
  q=re.search(r'\bid="([^"]+)"',attrs)
  blocks.append({'index':len(blocks),'tag':m.group(1),'id':q.group(1) if q else None,'text':txt})
 (W/'primary'/f'{name}.blocks.json').write_text(json.dumps(blocks,indent=2,ensure_ascii=False)+'\n')
 (W/'primary'/f'{name}.blocks.txt').write_text('\n\n'.join(f'[{b["index"]}] <{b["tag"]}> {b["id"] or ""}: {b["text"]}' for b in blocks)+'\n')
 print(name,len(blocks))
 for b in blocks:
  if b['tag'].startswith('h'):print(b['index'],b['text'])
