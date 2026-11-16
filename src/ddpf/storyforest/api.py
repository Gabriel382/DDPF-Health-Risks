"""Read-only first Studio integration; local API. LLM runs through CLI.

No .env secrets are sent to the browser. No fabricated demo results.
"""
from __future__ import annotations
from pathlib import Path
from .settings import Settings
from .storage import Store,forest

try:
    from fastapi import FastAPI,HTTPException
    from fastapi.responses import FileResponse
except ImportError as exc:
    raise RuntimeError('Install requirements-storyforest.txt to serve the Studio API') from exc

app=FastAPI(title='DDPF Story Forest local research API',version='0.1.0')
ROOT=Path(__file__).resolve().parents[3]

def db_path():return Settings.from_env(require_key=False).sqlite_path

@app.get('/api/health')
def health():
    s=Settings.from_env(require_key=False)
    return {'status':'ok','provider':s.provider,'model':s.litellm_model,'research_prototype':True}

@app.get('/api/events')
def events(limit:int=100,offset:int=0):
    with Store(db_path()) as store:
        return {'items':[e.to_dict() for e in store.list_events(offset=max(0,offset),limit=min(max(1,limit),200))],
                'total':store.count_events()}

@app.get('/api/links')
def links(method:str='full'):
    with Store(db_path()) as store:
        rows=store.links(method)
        return {'items':[l.to_dict() for l in rows[:2000]],'total':len(rows),'truncated':len(rows)>2000}

@app.get('/api/stories')
def stories(method:str='full'):
    with Store(db_path()) as store:
        count=store.count_events()
        if count>500:
            raise HTTPException(status_code=413,detail='Story view is limited to 500 events; add graph-aware pagination before loading larger experiments')
        result=forest(store.list_events(limit=500),store.links(method))
        return {'items':result,'total':len(result),'method':method}

@app.get('/api/evidence/{event_id}')
def evidence(event_id:str):
    with Store(db_path()) as store:
        event=store.get_event(event_id)
        if event is None:raise HTTPException(status_code=404,detail='Event not found')
        return event.to_dict()

@app.get('/storyforest')
def storyforest_ui():
    ui=ROOT/'interface'/'storyforest.html'
    if not ui.exists():raise HTTPException(status_code=404,detail='Extract overlay into DDPF repository root')
    return FileResponse(ui)

@app.get('/storyforest.js')
def storyforest_js():return FileResponse(ROOT/'interface'/'storyforest.js',media_type='text/javascript')

@app.get('/storyforest.css')
def storyforest_css():return FileResponse(ROOT/'interface'/'storyforest.css',media_type='text/css')
