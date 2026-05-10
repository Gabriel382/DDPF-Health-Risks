"""High-level Description module API."""

from pathlib import Path
from typing import Any

from rdflib import Graph

from ddpf.description.enhancers import ManualOntologyEnhancer
from ddpf.description.ontology_io import (
    load_ontology_graph,
    load_ontology_with_owlready,
    save_graph,
)


class DescriptionModule:
    """Reusable API for the DDPF Description module."""

    def __init__(
        self,
        propaphen_path: str | Path,
        domain_ontology_path: str | Path | None = None,
        network_ontology_path: str | Path | None = None,
    ) -> None:
        """Initialize the Description module.

        Args:
            propaphen_path: Path to the PropaPhen ontology.
            domain_ontology_path: Optional path to a domain ontology, such as UMLS.
            network_ontology_path: Optional path to a network ontology, such as WorldKG.
        """
        self.propaphen_path = Path(propaphen_path)
        self.domain_ontology_path = (
            Path(domain_ontology_path) if domain_ontology_path else None
        )
        self.network_ontology_path = (
            Path(network_ontology_path) if network_ontology_path else None
        )

        self.propaphen: Any | None = None
        self.propaphen_graph: Graph | None = None
        self.domain_graph: Graph | None = None
        self.network_graph: Graph | None = None

    def load(self, load_owlready: bool = True) -> "DescriptionModule":
        """Load PropaPhen and optional domain/network ontology resources.

        Args:
            load_owlready: If True, also load PropaPhen with Owlready2.
                External ontologies are loaded with rdflib to avoid Owlready2
                compatibility issues.

        Returns:
            The loaded DescriptionModule instance.
        """
        self.propaphen_graph = load_ontology_graph(self.propaphen_path)

        if load_owlready:
            self.propaphen = load_ontology_with_owlready(
                self.propaphen_path,
                ignore_imports=True,
            )

        if self.domain_ontology_path is not None:
            self.domain_graph = load_ontology_graph(self.domain_ontology_path)

        if self.network_ontology_path is not None:
            self.network_graph = load_ontology_graph(self.network_ontology_path)

        return self

    def enhance_manual(self, output_path: str | Path) -> Path:
        """Create a manually enhanced ontology by merging available ontology graphs.

        Args:
            output_path: Path where the enhanced ontology should be saved.

        Returns:
            Saved ontology path.
        """
        if self.propaphen_graph is None:
            raise RuntimeError("Call load() before enhance_manual().")

        enhancer = ManualOntologyEnhancer()
        enhanced_graph = enhancer.enhance(
            propaphen_graph=self.propaphen_graph,
            domain_graph=self.domain_graph,
            network_graph=self.network_graph,
        )

        return save_graph(enhanced_graph, output_path)

    def enhance_with_llm(
        self,
        llm: Any,
        text: str,
        output_path: str | Path,
        parent_class_name: str | None = None,
        max_concepts: int = 10,
    ) -> Path:
        """Create an LLM-assisted ontology enhancement output.

        Args:
            llm: LLM client implementing the DDPF LLM interface.
            text: Text used to suggest candidate ontology concepts.
            output_path: Path where the enhanced ontology should be saved.
            parent_class_name: Optional parent class used to guide concept extraction.
            max_concepts: Maximum number of candidate concepts requested from the LLM.

        Returns:
            Saved ontology path.
        """
        if self.propaphen_graph is None:
            raise RuntimeError("Call load() before enhance_with_llm().")

        parent_instruction = ""
        if parent_class_name is not None:
            parent_instruction = (
                f"The suggested concepts should be interpreted as candidates "
                f"under or related to the parent class: {parent_class_name}.\n"
            )

        prompt = (
            "Extract candidate ontology concepts from the following text.\n"
            f"{parent_instruction}"
            f"Return at most {max_concepts} concepts.\n"
            "Return one concept per line.\n"
            "Do not add explanations, numbering, or markdown.\n\n"
            f"{text}"
        )

        response = llm.generate(prompt)

        graph = Graph()
        graph += self.propaphen_graph

        saved_path = save_graph(graph, output_path)

        llm_output_path = Path(output_path).with_suffix(".llm.txt")
        llm_output_path.write_text(response, encoding="utf-8")

        return saved_path