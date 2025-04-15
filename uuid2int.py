import json
from pathlib import Path

INPUT_DIR = Path("/Users/mlenz/Downloads/scaling_requests/queries-aif")
PATTERN = "*.json"
OUTPUT_DIR = Path("data/queries/english/kialo-graphnli")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for input_path in INPUT_DIR.glob(PATTERN):
    with input_path.open() as f:
        graph = json.load(f)

    nodes_uuid2int = {
        node["nodeID"]: idx for idx, node in enumerate(graph["nodes"], start=1001)
    }
    edge_uuid2int = {
        edge["edgeID"]: idx for idx, edge in enumerate(graph["edges"], start=5001)
    }

    for node in graph["nodes"]:
        node["nodeID"] = nodes_uuid2int[node["nodeID"]]

    for edge in graph["edges"]:
        edge["edgeID"] = edge_uuid2int[edge["edgeID"]]
        edge["fromID"] = nodes_uuid2int[edge["fromID"]]
        edge["toID"] = nodes_uuid2int[edge["toID"]]

    output_path = OUTPUT_DIR / input_path.name

    with output_path.open("w") as f:
        json.dump(graph, f, indent=2)
