from __future__ import absolute_import, annotations

from dataclasses import dataclass

import numpy as np

from ..models.graph import Graph


@dataclass
class Result(object):
    """Store a graph and its similarity to the input query"""

    graph: Graph
    similarity: np.ndarray
    duration: float
