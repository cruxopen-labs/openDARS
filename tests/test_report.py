from dars.report import observed_bottlenecks
from dars.workload import ROOT


def test_bottlenecks_are_derived_and_published():
    lines = observed_bottlenecks()
    assert len(lines) == 4
    assert all(line.startswith("- ") for line in lines)
    # The checked-in report must agree with the measurements, so a data update
    # cannot silently leave stale prose behind.
    report = (ROOT / "reports/report.md").read_text()
    for line in lines:
        assert line in report


def test_decision_margins_tool_runs():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "tools/decision_margins.py"], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "headroom" in result.stdout
    assert "Gcycles" in result.stdout
