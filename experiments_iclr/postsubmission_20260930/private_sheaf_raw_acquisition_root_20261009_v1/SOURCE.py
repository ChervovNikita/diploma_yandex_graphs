from pathlib import Path
import socket,subprocess,json,os,urllib.request,hashlib,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'private_sheaf_raw_acquisition_root_20261009_v1';os.chdir(R)
assert socket.gethostname()=='anogena-2-0' and not A.exists()
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GNNM-research/1.0'}),timeout=30) as f:return f.read(50000000)
commit=json.loads(get('https://api.github.com/repos/yandex-research/heterophilous-graphs/commits/main'))['sha']
meta=json.loads(get('https://api.github.com/repos/yandex-research/heterophilous-graphs/contents/data/tolokers.npz?ref='+commit))
url='https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/'+commit+'/data/tolokers.npz';data=get(url)
assert len(data)==meta['size'] and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==meta['sha']
A.mkdir();(A/'tolokers.npz').write_bytes(data)
v=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_repository='yandex-research/heterophilous-graphs',source_commit=commit,source_blob=meta['sha'],source_url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),arrays_decoded=False,TEST_truth_scored=False,original_benchmark_exploratory=True)
(A/'ACQUISITION.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
