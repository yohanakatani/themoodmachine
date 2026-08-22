import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_with_empty_stdin(script_name):
    # The interactive loop in each script exits on the first blank line,
    # so a single newline on stdin lets the whole script run to completion.
    return subprocess.run(
        [sys.executable, script_name],
        cwd=ROOT,
        input="\n",
        capture_output=True,
        text=True,
        # Generous on purpose: the very first "import sklearn" in a brand
        # new virtualenv on Windows can take way longer than usual while
        # antivirus scans the freshly installed compiled dependencies.
        timeout=120,
    )


def test_main_runs_end_to_end():
    result = _run_with_empty_stdin("main.py")
    assert result.returncode == 0, result.stderr


def test_ml_experiments_runs_end_to_end():
    result = _run_with_empty_stdin("ml_experiments.py")
    assert result.returncode == 0, result.stderr
