from __future__ import absolute_import, annotations

import logging
import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Optional

import gensim
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_hub as hub
import torch

from ..libs.infersent import InferSent
from ..libs.sif import sif
from ..models.graph import Graph
from ..services.token_weighter import TokenWeighter
from ..services.utils import Config

logger = logging.getLogger("recap")
config = Config.get_instance()


class Embeddings(object):
    """Class to store embedding objects"""

    # Here will be the instance stored.
    _instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Embeddings._instance is None:
            Embeddings()

        for filename in config["embeddings_filter"]:
            Embeddings._instance[filename].load() # pyright: ignore

        return {
            key: value
            for key, value in Embeddings._instance.items() # pyright: ignore
            if key in config["embeddings_filter"]
        }

    @staticmethod
    def keys():
        return Embeddings._instance.keys() # pyright: ignore

    def __init__(self):
        """ Virtually private constructor. """
        if Embeddings._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Embeddings._instance = {
                filename: Embedding(filename)
                for filename in sorted(os.listdir(config["embeddings_folder"]))
                if filename.endswith(".model")
                or filename in config["embeddings_supervised"]
            }


@dataclass
class Embedding(object):
    filename: str
    embedding: Any = field(default=None, init=False)
    lowercase: bool = field(default=True, init=False)
    _tf_graph: tf.Graph = field(default=None, init=False)
    _tf_embedding: Any = field(default=None, init=False)
    _tf_session: tf.Session = field(default=None, init=False)
    _tf_text_input: Any = field(default=None, init=False)

    def _istype(self, obj: str) -> Optional[Any]:
        cls = None

        if obj == "doc2vec":
            cls = gensim.models.doc2vec.Doc2Vec
        elif obj == "infersent":
            cls = InferSent
        elif obj == "use":
            cls = hub.module.Module

        return isinstance(self.embedding, cls) # pyright: ignore

    def load(self) -> None:
        if self.embedding is None:
            logger.info(f"Loading embedding {self.filename} into memory.")

            filepath = os.path.join(config["embeddings_folder"], self.filename)

            if self.filename.endswith(".model"):
                self.embedding = gensim.models.KeyedVectors.load(filepath, mmap="r")
                if "I" in self or "Uhr" in self:
                    self.lowercase = False

            else:
                if "infersent" in self.filename:
                    self.embedding = InferSent(
                        {
                            "bsize": 64,
                            "word_emb_dim": 300,
                            "enc_lstm_dim": 2048,
                            "pool_type": "max",
                            "dpout_model": 0.0,
                            "version": 1,
                        }
                    )
                    self.embedding.load_state_dict(torch.load(filepath))
                    # CUDA does not work on macOS
                    # self.embedding = self.embedding.cuda()
                    self.embedding.set_w2v_path(
                        os.path.join(
                            config["embeddings_folder"], config["infersent_w2v_file"]
                        )
                    )

                elif "use-" in self.filename:
                    self._tf_graph = tf.Graph()
                    with self._tf_graph.as_default():
                        self._tf_text_input = tf.placeholder(
                            dtype=tf.string, shape=[None]
                        )
                        self.embedding = hub.Module(filepath)
                        self._tf_embedding = self.embedding(self._tf_text_input)
                        init_op = tf.group(
                            [tf.global_variables_initializer(), tf.tables_initializer()]
                        )
                    self._tf_graph.finalize()

                    self._tf_session = tf.Session(graph=self._tf_graph)
                    self._tf_session.run(init_op)

    def preprocess(self, cases: Dict[str, Graph], queries: Dict[str, Graph]):
        if self._istype("infersent"):
            tokens_nested = [graph.tokens for graph in cases.values()] + [
                graph.tokens for graph in queries.values()
            ]
            tokens_flat = list(chain.from_iterable(tokens_nested))

            self.embedding.build_vocab([" ".join(tokens_flat)], tokenize=False)

    def dimensionality(self) -> int:
        return self.embedding.vector_size

    def wmdistance(self, tokens1: List[str], tokens2: List[str]) -> float:
        return self.embedding.wmdistance(tokens1, tokens2)

    def query(self, tokens: List[str], token_weighter: TokenWeighter) -> np.ndarray:
        """Compute a vector based on a set of tokens

        For each operation defined, a vector will be generated.
        All vectors will then be concatenated.
        """

        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        vectors = []
        sentence_embedding = None

        for token in tokens:
            vector = self[token]
            if vector is not None:
                vectors.append(vector * token_weighter.index[token])

        if self._istype("doc2vec"):
            sentence_embedding = self.embedding.infer_vector(tokens)

        elif self._istype("infersent"):
            text = " ".join(tokens)
            sentence_embedding = self.embedding.encode([text])[0]

        elif self._istype("use"):
            text = " ".join(tokens)
            sentence_embedding = self._tf_session.run(
                self._tf_embedding, feed_dict={self._tf_text_input: [text]}
            )

        elif not vectors:
            logger.warning(
                "No word embeddings for tokens:\n{}".format(", ".join(tokens))
            )
            size = 0

            for operation in config["operations"]:
                size += self.dimensionality()

            sentence_embedding = np.zeros(size)
            vectors.append(np.zeros(self.dimensionality()))

        else:
            concat_embs: List[np.ndarray] = []

            for operation in config["operations"]:
                # The text pmean_ will be deleted so the number can be converted correctly
                operation = operation.replace("pmean", "")
                concat_embs.append(self.gen_mean(vectors, tokens, operation))

            sentence_embedding = np.concatenate(concat_embs, axis=0)

        return sentence_embedding, vectors

    def gen_mean(
        self, vals: List[np.ndarray], tokens: List[str], operation: str
    ) -> np.ndarray:
        """Calculate mean for given string

        The following values are allowed as operation:
            median
            mean
            gmean
            hmean
            max
            min
            any float
        """

        if operation == "median":
            return np.median(vals, axis=0)
        elif operation == "mean":
            return np.mean(vals, axis=0)
        elif operation == "gmean":
            return scipy.stats.gmean(vals, axis=0)
        elif operation == "hmean":
            return scipy.stats.hmean(vals, axis=0)
        elif operation == "max":
            return np.max(vals, axis=0)
        elif operation == "min":
            return np.min(vals, axis=0)
        elif operation == "sif":
            if self._istype("doc2vec"):
                return sif(vals, tokens, self.embedding.wv.vocab)
            else:
                return sif(vals, tokens, self.embedding.vocab)
        else:
            try:
                p = float(operation)
                return np.power(
                    np.mean(np.power(np.array(vals, dtype=complex), p), axis=0), 1 / p
                ).real
            except Exception:
                logger.error(f"Could not calculate mean for operation {operation}.")

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self:
            if self._istype("doc2vec"):
                return self.embedding.wv[key]

            else:
                return self.embedding[key]

        return None

    def __contains__(self, key: str) -> bool:
        if self._istype("doc2vec"):
            return key in self.embedding.wv

        elif self._istype("infersent") or self._istype("use"):
            return False

        else:
            return key in self.embedding
