"""Ontology enhancement strategies for the Description module."""

from rdflib import Graph


class ManualOntologyEnhancer:
    """Manual ontology enhancer based on RDF graph composition.

    This class reproduces the practical thesis logic at a reusable level:
    PropaPhen is kept as the core ontology, while domain and network resources
    are merged into an enhanced ontology graph.
    """

    def enhance(
        self,
        propaphen_graph: Graph,
        domain_graph: Graph | None = None,
        network_graph: Graph | None = None,
    ) -> Graph:
        """Merge PropaPhen with optional domain and network ontology graphs.

        Args:
            propaphen_graph: Core PropaPhen ontology graph.
            domain_graph: Optional domain ontology graph, such as UMLS.
            network_graph: Optional network ontology graph, such as WorldKG.

        Returns:
            Enhanced RDF graph.
        """
        enhanced_graph = Graph()

        enhanced_graph += propaphen_graph

        if domain_graph is not None:
            enhanced_graph += domain_graph

        if network_graph is not None:
            enhanced_graph += network_graph

        return enhanced_graph