"""Experimental GATE headless, bounded-memory document streaming.

Requires local Java + GATE distribution + preconfigured .xgapp (ANNIE alone
is NOT equivalent to the original biomedical/location gazetteer extraction).
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
from zipfile import ZipFile

EXTENSIONS={'.txt','.md','.csv'}

def documents_from_zip(path:str,max_bytes:int=2_000_000):
    with ZipFile(path) as archive:
        for info in archive.infolist():
            name=Path(info.filename)
            if info.is_dir() or name.suffix.lower() not in EXTENSIONS:continue
            if info.file_size>max_bytes:continue
            with archive.open(info) as fh:
                chunk=fh.read(max_bytes+1)
            if len(chunk)>max_bytes:continue
            yield info.filename,chunk.decode('utf-8',errors='replace')

def run_gate(zip_file:str,out_path:str,pipeline_path:str|None=None,
             max_doc_bytes:int=2_000_000):
    from gatenlp.gateworker import GateWorker
    pipeline_path=pipeline_path or os.getenv('GATE_PIPELINE_PATH')
    if not pipeline_path or not Path(pipeline_path).is_file():
        raise ValueError('Set GATE_PIPELINE_PATH to an EXISTING .xgapp pipeline; ANNIE alone != original DDPF Detection.')
    out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(str(out.with_suffix('.sqlite3')))
    db.execute('CREATE TABLE IF NOT EXISTS processed (hash TEXT PRIMARY KEY, filename TEXT NOT NULL)')
    db.commit()
    count=0
    worker=GateWorker(start=True,gatehome=os.getenv('GATE_HOME') or None)
    try:
        pipeline=worker.loadPipelineFromFile(str(Path(pipeline_path).resolve()))
        with out.open('a',encoding='utf-8') as output:
            for filename,text in documents_from_zip(zip_file,max_doc_bytes):
                hash_=hashlib.sha256(text.encode()).hexdigest()
                if db.execute('SELECT 1 FROM processed WHERE hash=?',(hash_,)).fetchone():continue
                gdoc=worker.createDocument(text)
                try:
                    worker.run4Document(pipeline,gdoc)
                    doc=worker.gdoc2pdoc(gdoc)
                    annotations=[]
                    for ann in doc.annset():
                        annotations.append({'type':ann.type,'start':ann.start,'end':ann.end,
                                            'features':dict(ann.features)})
                    row={'document_id':hash_,'filename':filename,'annotations':annotations,
                         'note':'Raw GATE annotations, NOT validated DDPF event observations'}
                    output.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
                    output.flush()
                    db.execute('INSERT INTO processed VALUES(?,?)',(hash_,filename));db.commit()
                    count+=1
                finally:
                    worker.deleteResource(gdoc)
    finally:
        worker.close();db.close()
    return count
