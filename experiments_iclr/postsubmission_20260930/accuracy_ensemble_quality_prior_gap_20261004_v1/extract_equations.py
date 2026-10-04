from extract_primary import Parser,walk,txt,P
import json
for key in ['shallow','camero']:
 a=Parser();a.feed((P/'sources'/f'{key}_html.html').read_text());out=[]
 for n in walk(a.root):
  if n.tag!='math':continue
  anc=n;ids=[];equation=False
  while anc:
   if anc.attrs.get('id'):ids.append(anc.attrs['id'])
   if 'ltx_equation' in anc.attrs.get('class',''):equation=True
   anc=anc.parent
  if n.attrs.get('display')=='block' or equation:
   out.append({'index':len(out),'ids':ids,'latex':n.attrs.get('alttext',txt(n))})
 (P/'sources'/f'{key}_equations.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(key,json.dumps(out,ensure_ascii=False,indent=2))
