
"""Inspect the existing Version20 benign-testing workflow."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK = (
    ROOT / "notebooks" / "kb"
    / "kb_flow_testing.ipynb"
)

KEYWORDS = [
    "benign",
    "known_test_row",
    "prepare_ae_context",
    "score_precomputed_flow",
    "add_behavioural_features",
]


def main():
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)

    with NOTEBOOK.open(encoding="utf-8") as file:
        notebook = json.load(file)

    cells = notebook.get("cells", [])

    matches = []

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        if any(
            keyword.lower() in source.lower()
            for keyword in KEYWORDS
        ):
            matches.append(index)

    print("=== BENIGN WORKFLOW INSPECTION ===")
    print("Matching cells:", matches)

    for index in matches:
        print(f"\n{'=' * 55}")
        print(f"CELL {index}")
        print("=" * 55)

        source = "".join(
            cells[index].get("source", [])
        )

        print(source)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
