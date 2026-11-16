"""Map a research-approved per-event CSV to Story Forest JSONL.

Does NOT fabricate provenance or split aggregated records automatically.
Needs event-level IDs, date, published_at and documentary evidence fields.
"""
import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('output');a=p.parse_args()
 required={'event_id','description','occurred_at','published_at','evidence_id','source_id','evidence_text'}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with open(a.csv,encoding='utf-8-sig',newline='') as inp, out.open('w',encoding='utf-8') as target:
  reader=csv.DictReader(inp)
  missing=required-set(reader.fieldnames or [])
  if missing:raise SystemExit('Missing required columns: '+', '.join(sorted(missing)))
  for row in reader:
   event={key:row[key] for key in ('event_id','description','occurred_at','published_at')}
   event['location']=row.get('location','')
   event['concept_ids']=[x.strip() for x in row.get('concept_ids','').split('|') if x.strip()]
   event['ontology_types']=[x.strip() for x in row.get('ontology_types','').split('|') if x.strip()]
   event['evidence']=[{'evidence_id':row['evidence_id'],'source_id':row['source_id'],
     'text':row['evidence_text'],'published_at':row['published_at'],'url':row.get('source_url','')}]
   target.write(json.dumps(event,ensure_ascii=False)+'\n')
 print('Wrote',out,'— audit duplicates and verify each event ID before evaluation.')
if __name__=='__main__':main()
