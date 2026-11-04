"""Export the authoritative notebook's code cells without running experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "notebooks/cnn-and-gan.ipynb").read_text(encoding="utf-8"))
sections = ["# Generated from notebooks/cnn-and-gan.ipynb by scripts/export_code.py.\n# Running this script trains all three tasks; use the notebook for selective execution.\n"]
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        sections.append("# %%\n" + "".join(cell["source"]).rstrip() + "\n")
target = ROOT / "src/cnn_and_gan.py"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(sections), encoding="utf-8", newline="\n")
print(f"Exported {target.relative_to(ROOT)}")
