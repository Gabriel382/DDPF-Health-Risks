"""Conservative, serializable domain models for retrospective event stories."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import date
from typing import Any

RELATIONS = ('same_event', 'documented_continuation', 'related_only', 'no_relation', 'unknown')

@dataclass
class Evidence:
    evidence_id: str
    source_id: str
    text: str
    published_at: str
    url: str = ''

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'Evidence':
        return cls(**{k:value.get(k, '') for k in ('evidence_id','source_id','text','published_at','url')})

@dataclass
class Event:
    event_id: str
    description: str
    occurred_at: str
    published_at: str
    location: str = ''
    concept_ids: list[str] = field(default_factory=list)
    ontology_types: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str,Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'Event':
        required = ('event_id', 'description', 'occurred_at', 'published_at')
        for key in required:
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f'Event requires nonempty {key}')
        for key in ('occurred_at', 'published_at'):
            date.fromisoformat(value[key][:10])
        return cls(event_id=value['event_id'], description=value['description'],
                   occurred_at=value['occurred_at'][:10], published_at=value['published_at'][:10],
                   location=value.get('location') or '',
                   concept_ids=list(value.get('concept_ids') or []),
                   ontology_types=list(value.get('ontology_types') or []),
                   evidence=[Evidence.from_dict(v) for v in (value.get('evidence') or [])],
                   metadata=value.get('metadata') or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Link:
    parent_id: str
    child_id: str
    relation: str
    evidence_ids: list[str]
    rationale: str
    score: float = 0.0  # model score is not calibrated probability
    method: str = ''
    status: str = 'proposed'
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Link':
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
