"""Description module for the DDPF package.

This module provides reusable utilities to specialize PropaPhen into a
domain-specific ontology through manual or LLM-assisted enhancement.
"""

from ddpf.description.enhancers import ManualOntologyEnhancer
from ddpf.description.module import DescriptionModule
from ddpf.description.ontology_io import (
    load_ontology_graph,
    load_ontology_with_owlready,
    save_graph,
)

__all__ = [
    "DescriptionModule",
    "ManualOntologyEnhancer",
    "load_ontology_graph",
    "load_ontology_with_owlready",
    "save_graph",
]