"""Prepare/check a separate Merlion environment; preserves the GEVIS NumPy stack."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    # Import this module only after setup; the bootstrap need not have NumPy.
    target=ROOT/'.cache/venvs/merlion'
    python=target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    worker=ROOT/'src/pecvd/experiments/merlion_worker.py'
    if not args.check:
        uv=shutil.which('uv')
        if uv is None: raise SystemExit('Install uv first: python -m pip install uv')
        if not python.is_file():
            subprocess.run([uv,'venv','--python','3.12',str(target)],cwd=ROOT,check=True)
        subprocess.run([uv,'pip','install','--python',str(python),'-r',str(ROOT/'config/requirements-merlion.txt')],cwd=ROOT,check=True)
    if not python.is_file(): raise SystemExit('Merlion environment is missing. Run this script without --check.')
    probe=subprocess.run([str(python),'-I',str(worker),'--probe'],capture_output=True)
    if probe.returncode: raise SystemExit('Merlion import check failed; run setup without --check.')
    print(probe.stdout.decode('utf-8'))
    return 0


if __name__=='__main__': raise SystemExit(main())
