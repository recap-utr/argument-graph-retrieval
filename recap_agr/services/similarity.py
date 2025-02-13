from __future__ import absolute_import, annotations

import logging
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np
import scipy
from nltk.metrics import edit_distance

from ..models.graph import Edge, Graph, Node
from ..models.nlp import Embeddings
from ..models.ontology import Ontology
from ..models.result import Result
from ..services import utils

logger = logging.getLogger("recap")

config = utils.Config.get_instance()


class Similarity(object):
    """Class to store similarity params and compute metrics based on them"""

    # Here will be the instance stored.
    _instance = None

    @staticmethod
    def get_instance() -> Similarity:
        """Static access method."""
        if Similarity._instance is None:
            Similarity()
        return Similarity._instance # pyright: ignore

    def __init__(self):
        """Virtually private constructor."""
        if Similarity._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Similarity._instance = self

    def graphs_ideal_similarity(
        self,
        graphs: Dict[str, Graph],
        query_graph: Graph,
        user_rankings: Dict[str, int],
    ) -> List[Result]:
        """Compute similarity between multiple graphs"""

        similarities: List[Result] = []
        max_ranking = max(user_rankings.values())

        for filename, user_ranking in user_rankings.items():
            similarities.append(
                Result(graphs[filename], user_ranking / max_ranking, 0.0)
            )

        return sorted(similarities, key=lambda result: result.similarity, reverse=True)

    def graphs_similarity(
        self, graphs: Dict[str, Graph], query_graph: Graph
    ) -> List[Result]:
        """Compute similarity between multiple graphs"""

        similarities: List[Result] = []

        for graph in graphs.values():
            start_time = timer()
            sim = self.graph_similarity(graph, query_graph)
            similarities.append(Result(graph, sim, timer() - start_time))

        return sorted(similarities, key=lambda result: result.similarity, reverse=True)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""

        sim = 0.0

        if vec1.any() and vec2.any():
            try:
                sim = 1 - scipy.spatial.distance.cosine(vec1, vec2)
            except Exception:
                pass

        return sim

    def _angular_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute angular similarity between two vectors"""

        sim = 0.0

        if vec1.any() and vec2.any():
            try:
                sim = (
                    1.0
                    - np.arccos(
                        np.dot(vec1, vec2)
                        / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    )
                    / np.pi
                )
            except Exception:
                pass

        return sim

    def _edit_distance(self, text1: str, text2: str) -> float:
        """Calculate the Levenshtein distance between two strings"""

        sim = 0

        if text1 == text2:
            sim = 1
        else:
            sim = 1 - (edit_distance(text1, text2) / max(len(text1), len(text2)))

        return sim

    def _word_movers_distance(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate the word mover's distance with gensim

        As multiple models are available, the results are averaged.
        """

        similarities = []

        embs = Embeddings.get_instance()
        for emb in embs.values():
            similarities.append(1 / (1 + emb.wmdistance(tokens1, tokens2)))

        return np.mean(similarities)

    def _fuzzify(self, s, u):
        """https://github.com/Babylonpartners/fuzzymax
        Sentence fuzzifier.
        Computes membership vector for the sentence S with respect to the
        universe U
        :param s: list of word embeddings for the sentence
        :param u: the universe matrix U with shape (K, d)
        :return: membership vectors for the sentence
        """
        f_s = np.dot(s, u.T)
        m_s = np.max(f_s, axis=0)
        m_s = np.maximum(m_s, 0, m_s)
        return m_s

    def _dynamax_jaccard(self, x, y):
        """https://github.com/Babylonpartners/fuzzymax
        DynaMax-Jaccard similarity measure between two sentences
        :param x: list of word embeddings for the first sentence
        :param y: list of word embeddings for the second sentence
        :return: similarity score between the two sentences
        """
        u = np.vstack((x, y))
        m_x = self._fuzzify(x, u)
        m_y = self._fuzzify(y, u)

        m_inter = np.sum(np.minimum(m_x, m_y))
        m_union = np.sum(np.maximum(m_x, m_y))
        return m_inter / m_union

    def _max_jaccard(self, x, y):
        """
        MaxPool-Jaccard similarity measure between two sentences
        :param x: list of word embeddings for the first sentence
        :param y: list of word embeddings for the second sentence
        :return: similarity score between the two sentences
        """
        m_x = np.max(x, axis=0)
        m_x = np.maximum(m_x, 0, m_x)
        m_y = np.max(y, axis=0)
        m_y = np.maximum(m_y, 0, m_y)
        m_inter = np.sum(np.minimum(m_x, m_y))
        m_union = np.sum(np.maximum(m_x, m_y))
        return m_inter / m_union

    def _fuzzy_jaccard(self, x, y):
        m_inter = np.sum(np.minimum(x, y))
        m_union = np.sum(np.maximum(x, y))
        return m_inter / m_union

    def _threshold_similarity(self, sim: float) -> float:
        """Adapt a similarity value to a specified threshold"""

        if sim < config["similarity_threshold"]:
            sim = 0.0
        elif config["similarity_threshold"] > 0:
            sim = (sim - config["similarity_threshold"]) / (
                1 - config["similarity_threshold"]
            )

        return sim

    def graph_similarity(self, graph1: Graph, graph2: Graph) -> float:
        """Compute similarity of two graphs based on their texts"""

        return self.general_similarity(graph1, graph2)

    def general_similarity(self, entity1: Any, entity2: Any) -> float:
        sim = 0.0

        if config["similarity_method"] == "edit":
            sim = self._edit_distance(entity1.text, entity2.text)

        elif config["similarity_method"] == "cosine":
            sim = self._cosine_similarity(entity1.vector, entity2.vector)

        elif config["similarity_method"] == "angular":
            sim = self._angular_similarity(entity1.vector, entity2.vector)

        elif config["similarity_method"] == "wmd":
            sim = self._word_movers_distance(entity1.tokens, entity2.tokens)

        elif config["similarity_method"] == "dynamax":
            sim = self._dynamax_jaccard(entity1.vectors, entity2.vectors)

        elif config["similarity_method"] == "maxpool":
            sim = self._max_jaccard(entity1.vectors, entity2.vectors)

        elif config["similarity_method"] == "fuzzy-jaccard":
            sim = self._fuzzy_jaccard(entity1.vector, entity2.vector)

        return self._threshold_similarity(sim)

    def node_similarity(self, node1: Node, node2: Node) -> float:
        """Compute similarity of nodes

        If it is an I-Node, the text will be compared

        If it is a RA-Node, the ontology can be used to compare the scheme
        """

        sim = 0.0

        if node1.type_ == node2.type_:
            if node1.type_ == "I":
                sim = self.general_similarity(node1, node2)

            elif config["use_schemes"]:
                if node1.text == node2.text:
                    sim = 1

                elif node1.type_ == "RA" and config["use_ontology"]:
                    ontology = Ontology.get_instance()
                    sim = self._threshold_similarity(
                        ontology.get_similarity(node1.text, node2.text)
                    )

            else:
                sim = 1

        # TODO: Check if this is correct and improves the result
        # elif node1.type_ in ["RA", "CA"] and node2.type_ in ["RA", "CA"]:
        #     sim = 0.5

        return sim

    def edge_similarity(self, edge1: Edge, edge2: Edge) -> float:
        """Compute edge similarity by comparing the four corresponding nodes"""

        return 0.5 * (
            self.node_similarity(edge1.from_node, edge2.from_node)
            + self.node_similarity(edge1.to_node, edge2.to_node)
        )
