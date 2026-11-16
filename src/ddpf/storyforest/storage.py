"""Persistent SQLite event/link store; bounded loading for incremental runs."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from .models import Event,Link

class Store:
    def __init__(self,path:str):
        self.path=path
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, published_at TEXT NOT NULL, body TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_events_pub ON events(published_at);
        CREATE TABLE IF NOT EXISTS links(parent_id TEXT NOT NULL,child_id TEXT NOT NULL,
          method TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(parent_id,child_id,method));
        CREATE TABLE IF NOT EXISTS progress(name TEXT PRIMARY KEY,value TEXT NOT NULL);
        ''')
        self.db.commit()

    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def has_event(self,event_id:str) -> bool:
        return self.db.execute('SELECT 1 FROM events WHERE id=?',(event_id,)).fetchone() is not None

    def insert_event(self,event:Event) -> bool:
        cur=self.db.execute('INSERT OR IGNORE INTO events VALUES(?,?,?)',
                            (event.event_id,event.published_at,json.dumps(event.to_dict(),ensure_ascii=False)))
        self.db.commit()
        return cur.rowcount==1

    def recent_events(self,before_or_at:str,limit:int=2000)->list[Event]:
        rows=self.db.execute('SELECT body FROM events WHERE published_at<=? ORDER BY published_at DESC,id DESC LIMIT ?',
                             (before_or_at,limit)).fetchall()
        return [Event.from_dict(json.loads(r['body'])) for r in rows]

    def list_events(self,offset:int=0,limit:int=100)->list[Event]:
        rows=self.db.execute('SELECT body FROM events ORDER BY published_at,id LIMIT ? OFFSET ?',
                             (min(500,limit),offset)).fetchall()
        return [Event.from_dict(json.loads(r['body'])) for r in rows]

    def add_link(self,link:Link):
        self.db.execute('INSERT OR REPLACE INTO links VALUES(?,?,?,?)',
                        (link.parent_id,link.child_id,link.method,json.dumps(link.to_dict(),ensure_ascii=False)))
        self.db.commit()

    def links(self,method:str|None=None)->list[Link]:
        if method:
            rows=self.db.execute('SELECT body FROM links WHERE method=? ORDER BY child_id,parent_id',(method,)).fetchall()
        else:
            rows=self.db.execute('SELECT body FROM links ORDER BY child_id,parent_id').fetchall()
        return [Link.from_dict(json.loads(row['body'])) for row in rows]

    def get_event(self,event_id:str)->Event|None:
        row=self.db.execute('SELECT body FROM events WHERE id=?',(event_id,)).fetchone()
        return Event.from_dict(json.loads(row['body'])) if row else None

    def count_events(self)->int:
        return self.db.execute('SELECT count(*) FROM events').fetchone()[0]

def forest(events:list[Event], links:list[Link])->list[dict]:
    """Build a forest from documented continuation edges only; unknowns stay roots.

    Only one parent per node; a duplicate/same-event relation is reported
    separately, never silently converted to temporal progression.
    """
    nodes={e.event_id:e for e in events}
    parent_of={}
    for link in sorted(links,key=lambda l:-l.score):
        if link.relation!='documented_continuation' or link.status!='accepted':continue
        if link.parent_id not in nodes or link.child_id not in nodes:continue
        if link.child_id==link.parent_id or link.child_id in parent_of:continue
        a=nodes[link.parent_id];b=nodes[link.child_id]
        if a.occurred_at>b.occurred_at:continue
        pointer=link.parent_id
        while pointer in parent_of:
            pointer=parent_of[pointer]
            if pointer==link.child_id:break
        else:
            parent_of[link.child_id]=link.parent_id
    children={id:[] for id in nodes}
    for child,parent in parent_of.items():children[parent].append(child)
    roots=[e for e in events if e.event_id not in parent_of]
    result=[]
    for root in roots:
        items=[];pending=[root.event_id]
        while pending:
            node=pending.pop(0)
            items.append({'event_id':node,'parent_id':parent_of.get(node),
                          'description':nodes[node].description,'occurred_at':nodes[node].occurred_at,
                          'location':nodes[node].location})
            pending.extend(sorted(children[node]))
        result.append({'story_id':'story-'+root.event_id,'root_id':root.event_id,'nodes':items})
    return result
