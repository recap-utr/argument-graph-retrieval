import json
from pathlib import Path

INPUT_DIR = Path("data/queries/english/microtexts-complex-arguebuf")
PATTERN = "*.json"
OUTPUT_DIR = Path("data/candidates/microtexts-complex")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for input_path in INPUT_DIR.glob(PATTERN):
    with input_path.open() as f:
        graph = json.load(f)

    original_ranking = graph["userdata"]["cbrEvaluations"][0]["ranking"]
    new_ranking: "dict[str, int]" = {}

    for name, rank in original_ranking.items():
        *parents, stem = name.split("/")
        new_ranking[stem + ".json"] = rank

    output_path = OUTPUT_DIR / input_path.name

    with output_path.open("w") as f:
        json.dump(
            {
                "candidates": list(new_ranking.keys()),
                "rankings": new_ranking,
            },
            f,
            indent=2,
        )
