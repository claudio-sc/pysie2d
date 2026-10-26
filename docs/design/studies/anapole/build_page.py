"""Assemble the "Dark Silicon Wires" showcase page from the template and the results.

Needs results/showcase.json, multipoles.json and showcase_500.json (showcase_data.py,
multipole_data.py, showcase_500_data.py). Writes results/dark-silicon-wires.html.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
html = (HERE / "showcase_template.html").read_text()
for key, name in (("__DATA__", "showcase.json"), ("__MP__", "multipoles.json"), ("__F5__", "showcase_500.json")):
    html = html.replace(key, (R / name).read_text())
(R / "dark-silicon-wires.html").write_text(html)
print(f"{len(html.encode()) / 1e6:.1f} MB")
