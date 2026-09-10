"""Execute named notebooks from clean kernels and export HTML with embedded outputs."""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("notebooks", nargs="+", help="Notebook filenames in notebooks/ or absolute paths")
args = parser.parse_args()
os.environ["IPYTHONDIR"] = str(ROOT / ".cache" / "ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".cache" / "jupyter-runtime")
for name in args.notebooks:
    path = ROOT / "notebooks" / name
    nb = nbformat.read(path, as_version=4)
    print("Executing:", path.name, flush=True)
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
    nbformat.validate(nb)
    codes = [c for c in nb.cells if c.cell_type == "code"]
    assert [c.execution_count for c in codes] == list(range(1, len(codes) + 1))
    assert not any(o.output_type == "error" for c in codes for o in c.outputs)
    html, _ = HTMLExporter().from_notebook_node(nb)
    path.with_suffix(".html").write_text(html, encoding="utf-8")
    print(f"PASS: {path.name}; {len(codes)} code cells, HTML exported.", flush=True)
