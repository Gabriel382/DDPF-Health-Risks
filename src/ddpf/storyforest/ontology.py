"""Conservative type-pair constraints, explicitly configured by the researcher.

NOT an OWL reasoner. The bundled .json contains no inferred disease-specific
assertions. When no explicit constraints match, validation remains unknown.
"""
from __future__ import annotations
import json
from pathlib import Path
from .models import Event

class OntologyGuard:
    def __init__(self, rules:dict|None=None):
        self.rules=rules or {}

    @classmethod
    def from_file(cls,path:str|Path|None):
        if not path:return cls({})
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def assess(self,parent:Event,child:Event,relation:str) -> tuple[bool,list[str]]:
        problems=[]
        if relation == 'documented_continuation' and parent.occurred_at > child.occurred_at:
            problems.append('backward_event_time')
        if relation == 'same_event':
            days=abs((__import__('datetime').date.fromisoformat(parent.occurred_at)-__import__('datetime').date.fromisoformat(child.occurred_at)).days)
            limit=int(self.rules.get('same_event_max_days',7))
            if days>limit: problems.append('same_event_too_far_apart')
        for pair in self.rules.get('forbidden_type_pairs',[]):
            if not (isinstance(pair,list) and len(pair)==2): continue
            if (pair[0] in parent.ontology_types and pair[1] in child.ontology_types) or (pair[1] in parent.ontology_types and pair[0] in child.ontology_types):
                problems.append(f'forbidden_type_pair:{pair[0]}:{pair[1]}')
        return not problems, problems
