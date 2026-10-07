import sys,json,hashlib,os,collections
sys.path.insert(0,os.path.abspath('gguf-py'))
from gguf import GGUFReader
p=sys.argv[1];r=GGUFReader(p,mode='r');h=hashlib.sha256()
with open(p,'rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
meta={'path':p,'file_bytes':os.path.getsize(p),'sha256':h.hexdigest(),'tensor_count':len(r.tensors),'payload_bytes':sum(t.n_bytes for t in r.tensors),'tensor_data_offset':min(t.data_offset for t in r.tensors)}
bytype=collections.Counter();cats=collections.Counter();rows=[]
for t in r.tensors:
 typ=int(t.tensor_type);bytype[typ]+=t.n_bytes;cat='ptq_gemv_candidate' if typ==143 and t.name!='token_embd.weight' else ('ptq_embedding_lookup' if typ==143 else 'small_or_non_ptq_weight');cats[cat]+=t.n_bytes;rows.append({'name':t.name,'shape':[int(x) for x in t.shape],'type':typ,'payload_bytes':t.n_bytes,'category':cat})
print(json.dumps({'file':meta,'bytes_by_tensor_type':dict(bytype),'categories_bytes':dict(cats),'tensors':rows},indent=2))
