"""Inventory already existing full-DDPF evidence. Do not fabricate original detection labels."""
from pathlib import Path
from .core import *

def run(root,out):
    out.mkdir(parents=True,exist_ok=True);base=root/'outputs'/'realdata';items=[];candidates=[]
    if base.exists():
        for p in sorted(base.rglob('*')):
            if not p.is_file() or 'reviewer_v8' in p.parts or 'reviewer_revision_v8' in p.parts:continue
            name=str(p.relative_to(root));low=name.lower()
            if p.suffix.lower() in ('.csv','.json','.md','.jsonl') and any(k in low for k in ('embedding_benchmark','structural','clustering','observation','detect','annotation','reference','ontology','competency')) and p.stat().st_size<25*1024*1024:
                items.append({'path':name,'sha256':sha(p),'bytes':p.stat().st_size})
                if p.suffix=='.csv' and ('observations' in p.name.lower() or 'detection' in low or 'annotation' in low):
                    try:
                        rows=csvread(p);cols=set(rows[0]) if rows else set();candidates.append({'path':name,'rows':len(rows),'columns':sorted(cols),'has_reference_column':bool(cols&{'gold_label','reference_label','target_label','manual_label','is_correct'}),'has_source_column':bool(cols&{'source','source_id','document_id','evidence','url'})})
                    except Exception:pass
    result={'status':'INVENTORY_COMPLETE','files':items,'candidate_observation_or_detection_tables':candidates,'note':'The integrated contribution includes PropaPhen as semantic foundation. This inventory does not claim new standalone ontology validation. Existing embedding baselines can be reused; detection quality requires original human reference labels and source units, not disease labels inferred from clustering.'}
    dump(out/'ddpf_evidence_inventory.json',result);return result
