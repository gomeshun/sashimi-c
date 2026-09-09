"""Execute every walkthrough cell in a fresh kernel and reject incomplete results."""
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient

source = Path(__file__).resolve().parents[1] / "notebooks/usage_walkthrough.ipynb"
notebook = nbformat.read(source, as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

kernel_name = notebook.metadata.get("kernelspec", {}).get("name", "python3")
with tempfile.TemporaryDirectory(prefix="walkthrough-execution-") as directory:
    NotebookClient(
        notebook,
        timeout=900,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": directory}},
    ).execute()

cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
if not cells:
    raise RuntimeError("No runnable examples")
if any(cell.execution_count is None for cell in cells):
    raise RuntimeError("Walkthrough contains an unexecuted code cell")
if any(output.output_type == "error" for cell in cells for output in cell.outputs):
    raise RuntimeError("Walkthrough contains an error output")

output = source.parents[1] / "artifacts/usage_walkthrough.executed.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, output)
print(f"Executed {len(cells)} code cells with kernel {kernel_name!r}: {output}")
