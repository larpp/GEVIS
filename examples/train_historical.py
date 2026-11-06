"""Run the preserved numeric/Fusion training program, without Jupyter.

Requires original training configuration and data. Existing reuse/overwrite
guards are retained. This is not an automatic generic-data training recipe.
"""
import argparse
import os
from pathlib import Path
import runpy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component', choices=['encoder', 'fusion'], required=True)
    p.add_argument('--root', type=Path, default=Path.cwd())
    args = p.parse_args()
    script = args.root.resolve() / 'heads/_legacy_training' / (args.component + '.py')
    if not script.is_file():
        raise FileNotFoundError('Build/install the optional FDC source release first')
    os.chdir(args.root.resolve())
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
