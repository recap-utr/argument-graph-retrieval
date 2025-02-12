from __future__ import absolute_import, annotations

import json
import logging
import os
import sys
import traceback
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import flask
import gensim
import nltk

from .models.graph import Graph
from .models.nlp import Embeddings
from .models.ontology import Ontology
from .models.result import Result
from .services import exporter, importer, retrieval, utils
from .services.evaluation import Evaluation
from .services.similarity import Similarity
from .services.token_weighter import TokenWeighter
from .services.vectorizer import Vectorizer
from .cli.texify import texify

logger = logging.getLogger("recap")
logger.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

config = utils.Config.get_instance()


def run() -> None:
    """Calculate similarity of queries and case base"""

    nltk.download("stopwords")
    nltk.download("punkt")
    Embeddings.get_instance()

    app = flask.Flask(
        config["app_name"], root_path=os.path.abspath(os.path.dirname(__file__))
    )
    app.config.update(TEMPLATES_AUTO_RELOAD=True, FLASK_ENV="development")
    app.secret_key = os.urandom(16)

    @app.route("/", methods=["POST", "GET"])
    def index():
        mac_results = None
        fac_results = None
        evaluation = None
        start_time = 0
        duration = 0
        eval_dict = {}
        query_file_name = ""
        embeddings = Embeddings.get_instance()

        if flask.request.method == "POST":
            try:
                # Check for file
                flask_query_file = flask.request.files.get("query-file")

                query_files = (
                    {flask_query_file.filename: flask_query_file.read()}
                    if flask_query_file
                    else {}
                )
                _update_config()
                embeddings = Embeddings.get_instance()

                if config["perform_mac"] or config["perform_fac"]:
                    similarity = Similarity.get_instance()
                    graphs = importer.jsonfiles2graphs()
                    token_weighter = TokenWeighter(graphs)

                    if not query_files:
                        filenames = []

                        for filename in sorted(os.listdir(config["queries_folder"])):
                            if filename.endswith(".json"):
                                with open(
                                        os.path.join(config["queries_folder"], filename),
                                        "r",
                                ) as file:
                                    query_files[filename] = file.read()

                                filenames.append(filename)

                        query_file_name = ", ".join(filenames)

                    else:
                        query_file_name = ", ".join(query_files.keys())

                    query_graphs = _get_query_graphs(query_files)

                    logger.debug("Preprocessing case-base with vectorizer.")
                    start_time = timer()

                    vectorizer = Vectorizer(
                        embeddings, token_weighter, graphs, query_graphs
                    )
                    _preprocess(graphs, vectorizer)
                    _preprocess(query_graphs, vectorizer)

                    evaluations: List[Evaluation] = []

                    for number_of_run in range(1, config["number_of_runs"] + 1):
                        logger.info(
                            f"Run {number_of_run} of {config['number_of_runs']}"
                        )

                        for query_graph in query_graphs.values():
                            mac = [Result(graph, 0.0, 0.0) for graph in graphs.values()]
                            fac = None
                            evaluation = None
                            query_start_time = timer()

                            if config["perform_mac"]:
                                mac = similarity.graphs_similarity(graphs, query_graph)
                                mac_results = exporter.get_results(mac)

                                if config["retrieval_limit"] > 0:
                                    mac = mac[: config["retrieval_limit"]]

                                evaluation = Evaluation(graphs, mac, query_graph, timer() - query_start_time)

                            if config["perform_fac"]:
                                fac = retrieval.fac(mac, query_graph)
                                fac_results = exporter.get_results(fac)

                                if config["retrieval_limit"] > 0:
                                    fac = fac[: config["retrieval_limit"]]

                                evaluation = Evaluation(graphs, fac, query_graph, timer() - query_start_time)

                            assert evaluation is not None
                            evaluations.append(evaluation)

                            if config["export_results"]:
                                exporter.export_results(
                                    query_graph.filename,
                                    mac_results,
                                    fac_results,
                                    evaluation,
                                )
                                flask.flash("Individual Results were exported.")

                    duration = (timer() - start_time) / config["number_of_runs"]
                    eval_dict = exporter.get_results_aggregated(evaluations)

                    if config["export_results_aggregated"]:
                        exporter.export_results_aggregated(
                            eval_dict, duration, **config._config
                        )
                        if config["texify"]:
                            texify()
                            logger.info("Copied TeX code to clipboard.")
                        flask.flash("Aggregated Results were exported.")

                    if len(query_graphs) > 1:
                        mac_results = ""
                        fac_results = ""
                else:
                    flask.flash("No operation (MAC/FAC) has been selected", "warning")
            except:
                flask.flash(traceback.format_exc(), "error")

        return flask.render_template(
            "index.html",
            config=config,
            embeddings=Embeddings.keys(),
            query_file_name=query_file_name,
            fac_results=fac_results,
            mac_results=mac_results,
            eval_dict=eval_dict,
            duration=duration,
        )

    app.run(host=config["flask_host"], port=config["flask_port"])


def _get_query_graphs(files: Dict[str, str]) -> Dict[str, Graph]:
    """Convert json query to graph"""

    query_graphs = {}

    for key, value in files.items():
        query_json = json.loads(value)
        query_graph = importer.jsonobj2graph(query_json, key)

        query_graphs[query_graph.filename] = query_graph

    return query_graphs


def _preprocess(graphs: Dict[str, Graph], vectorizer: Vectorizer) -> None:
    """Create graph vectors for all input files"""
    for graph in graphs.values():
        vectorizer.get_graph_vectors(graph)


def _update_config() -> None:
    config["operations"] = [
        x.strip() for x in flask.request.form["operations"].split(",")
    ]
    config["similarity_threshold"] = float(flask.request.form["similarity-threshold"])
    config["similarity_method"] = flask.request.form["similarity-method"]
    config["token_weighting"] = flask.request.form["token-weighting"]
    config["retrieval_limit"] = int(flask.request.form["prefilter-limit"])
    config["stemming"] = bool(flask.request.form.get("stemming", False))
    config["ignore_stopwords"] = bool(flask.request.form.get("ignore-stopwords", False))
    config["export_results"] = bool(flask.request.form.get("export-results", False))
    config["export_results_aggregated"] = bool(
        flask.request.form.get("export-results-aggregated", False)
    )
    config["a_star_queue_limit"] = int(flask.request.form["a-star-queue-limit"])
    config["perform_fac"] = bool(flask.request.form.get("perform-fac", False))
    config["perform_mac"] = bool(flask.request.form.get("perform-mac", False))
    config["use_schemes"] = bool(flask.request.form.get("use-schemes", False))
    config["use_ontology"] = bool(flask.request.form.get("use-ontology", False))
    config["number_of_runs"] = int(flask.request.form["number-of-runs"])

    config["casebase_folder"] = flask.request.form["casebase-folder"]
    config["queries_folder"] = flask.request.form["queries-folder"]
    config["candidates_folder"] = flask.request.form["candidates-folder"]
    config["results_folder"] = flask.request.form["results-folder"]

    config["embeddings_filter"] = [
        emb for emb in flask.request.form.getlist("embeddings-filter")
    ]
