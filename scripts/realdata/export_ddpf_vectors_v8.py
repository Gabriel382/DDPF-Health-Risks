"""Convert a DDPF NumPy matrix into ID-keyed JSONL; never infer row order."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',required=True);p.add_argument('--ids',required=True,help='CSV with observation_id column in exact matrix row order, or JSONL with observation_id and optional row');p.add_argument('--output',required=True);a=p.parse_args();ids=Path(a.ids)
    with ids.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f)) if ids.suffix=='.csv' else [json.loads(x) for x in f if x.strip()]
    if rows and all('row' in r and r['row'] not in (None,'') for r in rows):
        rows=sorted(rows,key=lambda r:int(r['row']));assert [int(r['row']) for r in rows]==list(range(len(rows))),'invalid row indices'
    names=[r['observation_id'] for r in rows];x=np.load(a.matrix,allow_pickle=False)
    if not isinstance(x,np.ndarray) or x.ndim!=2 or len(x)!=len(names) or len(set(names))!=len(names) or not np.isfinite(x).all():raise ValueError('matrix/ID mismatch or non-finite embeddings')
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    with dest.open('w',encoding='utf8') as f:
        for id,v in zip(names,x):f.write(json.dumps({'observation_id':id,'vector':v.tolist()})+'\n')
    print(f'{len(names)} ID-keyed vectors, {x.shape[1]} dimensions: {dest}')
if __name__=='__main__':main()
