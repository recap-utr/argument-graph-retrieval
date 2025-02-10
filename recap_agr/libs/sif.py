from __future__ import absolute_import, annotations

from typing import Dict, List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict

from ..services.utils import Config

config = Config.get_instance()


def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i, :] = w[i, :].dot(We[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def compute_pc(X: np.ndarray, npc: int = 1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X: np.ndarray, npc: int = 1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def sif(vectors: List[np.ndarray], tokens: List[str], words: List[str], rmpc: int = 1):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """

    We = np.array(vectors).T

    # word2weight['str'] is the weight for the word 'str'
    word2weight = SifWeights.get_instance()

    # weight4ind[i] is the weight for the i-th word
    weight4ind = get_weight(tokens, word2weight)

    # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    x, m, _ = sentences2idx(" ".join(tokens), words)

    w = seq2weight(x, m, weight4ind)  # get word weights

    emb = get_weighted_average(We, x, w)

    if rmpc > 0:
        emb = remove_pc(emb, rmpc)

    return emb


class SifWeights:
    """Store general application settings as a singleton"""

    # Here will be the instance stored.
    _instance = None
    weights: Dict[str, float]

    @staticmethod
    def get_instance():
        """ Static access method. """
        if SifWeights._instance is None:
            SifWeights()
        return SifWeights._instance

    def __init__(self):
        """ Virtually private constructor. """
        if SifWeights._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SifWeights._instance = self

            if (
                config["sif_parameter"] <= 0
            ):  # when the parameter makes no sense, use unweighted
                config["sif_parameter"] = 1.0

            word2weight = defaultdict(lambda: 1)
            with open(config["sif_weightfile"]) as f:
                lines = f.readlines()
            N = 0
            for i in lines:
                i = i.strip()
                if len(i) > 0:
                    i = i.split()
                    if len(i) == 2:
                        word2weight[i[0]] = float(i[1])
                        N += float(i[1])
                    else:
                        print(i)
            for key, value in word2weight.items():
                word2weight[key] = config["sif_parameter"] / (
                    config["sif_parameter"] + value / N
                )

            self.weights = word2weight

    def items(self):
        return self.weights.items()

    def __getitem__(self, key: str):
        if key in self.weights:
            return self.weights[key]
        return None


def get_weight(words, word2weight):
    weight4ind = {}
    for ind, word in enumerate(words):
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq1 = []
    for i in sentences:
        seq1.append(getSeq(i, words))
    x1, m1 = prepare_data(seq1)
    return x1, m1


def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype("float32")
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype="float32")
    return weight
