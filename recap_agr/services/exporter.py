from __future__ import absolute_import, annotations

import csv
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.result import Result
from ..services import utils
from ..services.evaluation import Evaluation

logger = logging.getLogger("recap")
config = utils.Config.get_instance()


def get_results(results: List[Result]) -> List[Dict[str, Any]]:
    """Convert the results to strings"""

    query_result = []

    for i, result in enumerate(results):
        query_result.append(
            {
                "name": result.graph.filename,
                "rank": i + 1,
                "similarity": np.around(result.similarity, 3),
                "text": result.graph.text,
                "duration": result.duration,
            }
        )

    return query_result


def export_results(
    query_file_name: str,
    mac_results: Optional[List[Dict[str, Any]]],
    fac_results: Optional[List[Dict[str, Any]]],
    evaluation: Evaluation,
) -> None:
    """Write the results to csv files

    The files will have mac, fac and eval appended to differentiate.
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    filename = os.path.join(
        config["results_folder"], "{}-{}".format(timestamp, query_file_name)
    )
    fieldnames = ["name", "rank", "similarity", "text"]

    if not os.path.exists(config["results_folder"]):
        os.makedirs(config["results_folder"])

    if mac_results:
        with open("{}-mac.csv".format(filename), "w", newline="") as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames)
            csvwriter.writeheader()
            csvwriter.writerows(mac_results)

    if fac_results:
        with open("{}-fac.csv".format(filename), "w", newline="") as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames)
            csvwriter.writeheader()
            csvwriter.writerows(fac_results)

    eval_dict = evaluation.as_dict()
    with open("{}-eval.csv".format(filename), "w", newline="") as csvfile:
        csvwriter = csv.DictWriter(csvfile, ["metric", "value"])
        csvwriter.writeheader()

        if "unranked" in eval_dict:
            for key, value in eval_dict["unranked"].items():
                csvwriter.writerow({"metric": key, "value": value})

        if "ranked" in eval_dict:
            for key, value in eval_dict["ranked"].items():
                csvwriter.writerow({"metric": key, "value": value})


def get_results_aggregated(
    evaluations: List[Evaluation]
) -> Dict[str, Dict[str, float]]:
    """Return multiple evaluations as an aggregated dictionary."""

    ranked_aggr: Dict[str, float] = defaultdict(float)
    unranked_aggr: Dict[str, float] = defaultdict(float)

    for evaluation in evaluations:
        eval_dict = evaluation.as_dict()

        if "unranked" in eval_dict:
            for key, value in eval_dict["unranked"].items():
                unranked_aggr[key] += value

        if "ranked" in eval_dict:
            for key, value in eval_dict["ranked"].items():
                ranked_aggr[key] += value

    eval_dict_aggr = {"unranked": unranked_aggr, "ranked": ranked_aggr}

    for eval_type in eval_dict_aggr.values():
        for key, value in eval_type.items():
            eval_type[key] = round((value) / len(evaluations), 3)

    return eval_dict_aggr


def export_results_aggregated(
    evaluation: Dict[str, Dict[str, float]], duration: float, **kwargs
) -> None:
    """Write the results to file"""

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    filename = os.path.join(config["results_folder"], timestamp)

    if not os.path.exists(config["results_folder"]):
        os.makedirs(config["results_folder"])

    with open(f"{filename}.json", "w") as outfile:
        tex_values = []

        for eval_type in evaluation.values():
            for value in eval_type.values():
                tex_values.append(r"\(" + str(round(value, 3)) + r"\)")

        tex_str = r"& {} \\".format(" & ".join(tex_values))

        json_out = {
            "Results": evaluation,
            "Duration": round(duration, 3),
            "Parameters": kwargs,
        }
        json.dump(json_out, outfile, indent=4, ensure_ascii=False)
