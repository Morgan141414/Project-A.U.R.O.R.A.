"""One-command launcher for the A.U.R.O.R.A. local interface."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
python = ROOT / ".venv" / "bin" / "python"
if not python.exists():
    print("Missing .venv. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")
    raise SystemExit(1)
subprocess.run([str(python), str(ROOT / "app.py")], cwd=ROOT, check=False)
