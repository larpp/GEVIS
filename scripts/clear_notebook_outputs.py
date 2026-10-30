"""Clear notebook execution state with explicit UTF-8 JSON handling."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def clear(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        clear(Path(argument))
