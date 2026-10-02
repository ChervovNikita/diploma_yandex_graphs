"""Preserve exact documentary passages and render source pages, never scientific code."""
from pathlib import Path
import json,hashlib,subprocess
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'SEAL.json').exists()
for name in ['evidence','renders']:(ROOT/name).mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
specs=[('BC1','bayesian_cp',[4,5],'Tempered GDC, probability mean, NLL conformal score'),('BC2','bayesian_cp',[12,13],'Fixed shared model/mask samples and native recipe'),('SN1','snaps',[5,6],'Feature-neighbor scores and conditional efficiency premises'),('SN2','snaps',[7],'Source hyperparameter/calibration split recipe'),('SP1','sparsification',[3,4],'Learned layerwise edge removal and CP training quantile'),('SP2','sparsification',[11],'Native architecture and parameter-selection recipe'),('BG1','bgcn',[3,4],'Joint graph/model sampling, MAP graph approximation and algorithm'),('BG2','bgcn',[5],'Native classification recipe'),('GP1','gpn_resolved',[5,6],'Density pseudo-count PPR and uncertainty-directed propagation'),('GP2','gpn_resolved',[19,20],'Native uncertainty summaries and training recipe')]
passages=[]
for key,source,pages,purpose in specs:
 pdf=ROOT/'primary'/f'{source}.pdf';text=ROOT/'primary'/f'{source}.txt';reader=PdfReader(pdf);out=ROOT/'evidence'/f'{key}_{source}.txt';out.write_text('\n'.join(f'[PDF PAGE {i}]\n{reader.pages[i-1].extract_text()}' for i in pages))
 passages.append(dict(key=key,source=source,pdf_pages=pages,purpose=purpose,pdf_path=str(pdf.relative_to(ROOT)),pdf_sha256=sha(pdf),source_text_path=str(text.relative_to(ROOT)),source_text_sha256=sha(text),passage_path=str(out.relative_to(ROOT)),passage_sha256=sha(out)))
(ROOT/'evidence/PASSAGE_INDEX.json').write_text(json.dumps(passages,indent=2)+'\n')
render_specs=[('bayesian_cp',4),('bayesian_cp',5),('bayesian_cp',12),('bayesian_cp',13),('snaps',5),('snaps',6),('snaps',7),('sparsification',3),('sparsification',4),('sparsification',11),('bgcn',3),('bgcn',4),('bgcn',5),('gpn_resolved',5),('gpn_resolved',6),('gpn_resolved',20)]
records=[]
for source,page in render_specs:
 prefix=ROOT/'renders'/f'{source}_page{page}'
 subprocess.run(['/Users/alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm','-f',str(page),'-l',str(page),'-singlefile','-r','110','-png',str(ROOT/'primary'/f'{source}.pdf'),str(prefix)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 p=prefix.with_suffix('.png');records.append(dict(source=source,pdf_page=page,path=str(p.relative_to(ROOT)),sha256=sha(p),visually_inspected=False))
for i in range(0,len(records),2):
 images=[Image.open(ROOT/r['path']).convert('RGB') for r in records[i:i+2]]
 canvas=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)+35),'white');draw=ImageDraw.Draw(canvas);x=0
 for im,r in zip(images,records[i:i+2]):canvas.paste(im,(x,35));draw.text((x+8,8),f"{r['source']} PDF page {r['pdf_page']}",fill='black');x+=im.width
 canvas.save(ROOT/'renders'/f'contact_{i//2+1}.png')
(ROOT/'renders/RENDER_INDEX.json').write_text(json.dumps(records,indent=2)+'\n')
print('Preserved10 passage groups,16 full page renders,8 contacts; visual inspection pending.')
