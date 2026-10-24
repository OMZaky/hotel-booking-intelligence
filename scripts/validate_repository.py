"""Offline checks for recipe validity and clean, syntactically valid notebook sources."""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hotel_ml.experiments.config import load_config  # noqa: E402


def main() -> None:
    configs = sorted((ROOT / "configs").glob("*/*/*.json"))
    if not configs:
        raise ValueError("No experiment configs found")
    for path in configs:
        config = load_config(path)
        if path.parent.name != config["version"] or path.parent.parent.name != config["task"]:
            raise ValueError(f"Recipe location disagrees with its task/version: {path}")
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise ValueError("No notebooks found")
    code_cells = 0
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook["nbformat"] != 4:
            raise ValueError(f"Unsupported notebook format: {path}")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    raise ValueError(f"Clear outputs and execution counts: {path}, cell {index}")
                ast.parse("".join(cell["source"]), filename=f"{path}:cell{index}")
                code_cells += 1
    print(
        f"Validated {len(configs)} recipes, {len(notebooks)} notebook(s), {code_cells} code cells."
    )


if __name__ == "__main__":
    main()
