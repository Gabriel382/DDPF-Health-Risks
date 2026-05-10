"""Ontology input/output utilities for the DDPF Description module."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from rdflib import Graph, OWL


def _require_owlready2() -> Any:
    """Import Owlready2 only when Owlready2 ontology loading is required."""
    try:
        import owlready2
    except ImportError as exc:
        raise ImportError(
            "Owlready2 is required for Owlready2-based ontology operations."
        ) from exc

    return owlready2


def strip_owl_imports(ontology_path: str | Path) -> Path:
    """Create a temporary ontology copy without owl:imports.

    This avoids remote ontology loading during local demos and tests.
    """
    ontology_path = Path(ontology_path)

    graph = Graph()
    graph.parse(ontology_path)

    for triple in list(graph.triples((None, OWL.imports, None))):
        graph.remove(triple)

    temporary_file = NamedTemporaryFile(
        mode="wb",
        suffix=".owl",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    graph.serialize(destination=str(temporary_path), format="xml")
    return temporary_path


def load_ontology_with_owlready(
    path_or_iri: str | Path,
    ignore_imports: bool = True,
) -> Any:
    """Load an ontology with Owlready2.

    Use this mainly for PropaPhen or small ontologies known to be compatible
    with Owlready2.
    """
    owlready2 = _require_owlready2()

    value = str(path_or_iri)
    is_remote = value.startswith(("http://", "https://"))

    if ignore_imports and not is_remote:
        sanitized_path = strip_owl_imports(path_or_iri)

        ontology = owlready2.get_ontology(
            f"http://ddpf.local/ontology/{sanitized_path.stem}.owl"
        )

        with sanitized_path.open("rb") as file_object:
            ontology.load(fileobj=file_object)

        return ontology

    return owlready2.get_ontology(value).load()


def load_ontology_graph(path: str | Path) -> Graph:
    """Load an ontology as an RDF graph.

    This is safer for external ontologies such as WorldKG, where the same IRI
    may be used in ways that Owlready2 rejects.
    """
    graph = Graph()
    graph.parse(str(path))
    return graph


def save_graph(graph: Graph, output_path: str | Path, rdf_format: str = "xml") -> Path:
    """Save an RDF graph to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph.serialize(destination=str(output_path), format=rdf_format)
    return output_path