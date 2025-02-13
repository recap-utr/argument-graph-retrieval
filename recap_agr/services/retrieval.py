from __future__ import absolute_import, annotations

import bisect
import logging
import multiprocessing
import random
from typing import List, Tuple, Union, Optional
from timeit import default_timer as timer

from ..models.graph import Edge, Graph, Node
from ..models.mapping import SearchNode
from ..models.result import Result
from ..services import utils
from ..services.similarity import Similarity

logger = logging.getLogger("recap")
config = utils.Config.get_instance()


def fac(mac_results: List[Result], query_graph: Graph) -> List[Result]:
    """Perform an in-depth analysis of the prefilter results"""

    results: List[Result] = []
    params = [
        (mac_result.graph, query_graph, i, len(mac_results))
        for i, mac_result in enumerate(mac_results)
    ]

    logger.info(f"A* Search for query '{query_graph.filename}'.")

    if config["debug"]:
        results = [a_star_search(*param) for param in params]
    else:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(a_star_search, params)

    results.sort(key=lambda result: result.similarity, reverse=True)

    return results


# According to Bergmann and Gil, 2014
def a_star_search(
    case_graph: Graph, query_graph: Graph, current_iteration: int, total_iterations: int
) -> Result:
    """Perform an A* analysis of the case base and the query"""

    start_time = timer()
    q: List[SearchNode] = []
    s0 = SearchNode(query_graph.all_nodes, query_graph.edges)

    bisect.insort(q, s0)

    while q[-1].nodes or q[-1].edges:
        q = _expand(q, case_graph, query_graph)

    candidate = q[-1]

    logger.debug(
        f"A* search for {case_graph.filename} finished. ({current_iteration}/{total_iterations})"
    )

    return Result(
        case_graph,
        candidate.mapping.get_similarity(
            len(query_graph.all_nodes), len(query_graph.edges)
        ),
        timer() - start_time,
    )


def _expand(
    q: List[SearchNode], case_graph: Graph, query_graph: Graph
) -> List[SearchNode]:
    """Expand a given node and its queue"""

    s = q[-1]
    mapped = False
    key_q, x_q, iterator = select1(s, query_graph, case_graph)

    if key_q and x_q and iterator:
        for x_c in iterator:
            if s.mapping.is_legal_mapping(x_q, x_c):
                s_new = SearchNode(s.nodes, s.edges, s.mapping)
                s_new.mapping.map(x_q, x_c)
                s_new.remove(key_q, x_q)
                s_new.f = g(s_new, query_graph) + h2(s_new, query_graph, case_graph)
                bisect.insort(q, s_new)
                mapped = True

        if mapped:
            q.remove(s)
        else:
            s.remove(key_q, x_q)

    return (
        q[len(q) - config["a_star_queue_limit"] :]
        if config["a_star_queue_limit"] > 0
        else q
    )


def select1(s: SearchNode, query_graph: Graph, case_graph: Graph) -> Tuple:
    key_q = None
    x_q: Union[Node, Edge, None] = None
    iterator: Optional[List[Union[Node, Edge]]] = None

    if s.nodes:
        key_q, x_q = random.choice(list(s.nodes.items()))
        iterator = list(
            case_graph.i_nodes.values()
            if x_q.type_ == "I"
            else case_graph.s_nodes.values()
        )
    elif s.edges:
        key_q, x_q = random.choice(list(s.edges.items()))
        iterator = list(case_graph.edges.values())

    return key_q, x_q, iterator


def select2(s: SearchNode, query_graph: Graph, case_graph: Graph) -> None:
    pass


def h1(s: SearchNode, query_graph: Graph, case_graph: Graph) -> float:
    """Heuristic to compute future costs"""

    return (len(s.nodes) + len(s.edges)) / (
        len(query_graph.all_nodes) + len(query_graph.edges)
    )


def h2(s: SearchNode, query_graph: Graph, case_graph: Graph) -> float:
    h_val = 0
    sim = Similarity.get_instance()
    x: Union[Node, Edge]
    y: Union[Node, Edge]

    for x in s.nodes.values():
        max_sim = 0

        for y in (
            case_graph.i_nodes.values()
            if x.type_ == "I"
            else case_graph.s_nodes.values()
        ):
            current_sim = sim.node_similarity(x, y)
            if current_sim > max_sim:
                max_sim = current_sim

        h_val += max_sim

    for x in s.edges.values():
        max_sim = 0

        for y in case_graph.edges.values():
            current_sim = sim.edge_similarity(x, y)
            if current_sim > max_sim:
                max_sim = current_sim

        h_val += max_sim

    return h_val / (len(query_graph.all_nodes) + len(query_graph.edges))


def g(s: SearchNode, query_graph: Graph) -> float:
    """Function to compute the costs of all previous steps"""

    return s.mapping.get_similarity(len(query_graph.all_nodes), len(query_graph.edges))
