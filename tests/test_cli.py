import json
from pathlib import Path
import subprocess
import sys


def test_module_cli_entrypoints(tmp_path):
    root = Path(__file__).resolve().parents[1]
    space = tmp_path / "space.json"
    space.write_text(json.dumps(dict(hardware=dict(vpu_lanes=[4]), mappings=[dict(chunk=4)])))
    commands = [
        ["calibrate"], ["analyze"],
        ["search", "--method", "grid", "--target", "fpga", "--space", str(space)],
        ["search", "--method", "nsga2", "--target", "asic", "--space", str(space),
         "--population", "4", "--generations", "1"],
        ["report", "--no-search"],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, "-m", "dars.cli", *command, "--out", str(tmp_path)],
                                cwd=root, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
    assert (tmp_path / "report.md").exists()
    for stem in ("fpga_grid", "asic_nsga2"):
        data = json.loads((tmp_path / (stem + ".json")).read_text())
        assert data["evaluated"] == 1
        assert (tmp_path / (stem + ".png")).stat().st_size > 1000
