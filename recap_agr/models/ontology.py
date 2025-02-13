from __future__ import absolute_import, annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ruamel import yaml

from ..services.utils import Config

config = Config.get_instance()


@dataclass
class OntologyNode(object):
    """Store values for one node of the ontology"""

    value: str
    depth: int
    parent: Optional[OntologyNode]
    children: List[OntologyNode] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.append(self)


class Ontology(object):
    """Store ontology as singleton"""

    _instance = None
    nodes: Dict[str, OntologyNode] = {}
    root_node: Optional[OntologyNode]

    @staticmethod
    def get_instance() -> Ontology:
        """ Static access method. """
        if Ontology._instance is None:
            Ontology()
        return Ontology._instance # pyright: ignore

    def __init__(self) -> None:
        """ Virtually private constructor. """
        if Ontology._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Ontology._instance = self
            with open("ontology.yml", "r") as f:
                self._load_ontology(yaml.safe_load(f))

    def _load_ontology(
        self, nodes_dict: Dict[str, Any], parent: Optional[OntologyNode] = None, depth: int = 0
    ) -> None:
        if nodes_dict is not None:
            current_node = self.add_node(nodes_dict["val"], depth, parent)

            if self.root_node is None and parent is None:
                self.root_node = current_node

            children = nodes_dict.get("children")
            if children:
                for child in children:
                    self._load_ontology(child, current_node, depth + 1)

    def add_node(
        self, value: str, depth: int, parent: Optional[OntologyNode] = None
    ) -> OntologyNode:
        """Add a new node to the ontology"""

        node_value = value.lower().replace("argument from ", "")

        new_node = OntologyNode(node_value, depth, parent)
        self.nodes[value] = new_node
        return new_node

    def get_similarity(self, scheme1: str, scheme2: str) -> float:
        """Traverse the ontology tree and return the similarity of the common parent leaf"""

        node1 = self.nodes.get(scheme1.lower(), self.root_node)
        node2 = self.nodes.get(scheme2.lower(), self.root_node)

        n_1 = node1.depth if node1 else 0
        n_2 = node2.depth if node2 else 0

        while node1 != node2 and node1 is not None and node2 is not None:
            if node1.depth > node2.depth:
                node1 = node1.parent
            else:
                node2 = node2.parent

        return self._wu_palmer(node1.depth, n_1, n_2) if node1 is not None else 0.0

    def _wu_palmer(self, n_x: int, n_1: int, n_2: int):
        """Calculate similarity automatically based on Wu/Palmer.

        The function has an upper bound of 1.
        """

        return (2 * n_x) / (n_1 + n_2)
