"""Use the pinned optimizer launcher with only its model entrypoint substituted."""
import argparse
from pathlib import Path
import runpy
import subprocess
import sys
from sobol_repair import ROOT, TOOLS, verify


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sobol-repair', required=True)
    args, remaining = parser.parse_known_args()
    verify(args.sobol_repair)
    original = subprocess.Popen

    def popen(command, *positional, **keywords):
        if (isinstance(command, list) and len(command) > 1
                and command[1] == str(TOOLS/'peptide_rgpe_merged_worker.py')):
            command = [command[0], str(ROOT/'repair_worker.py'),
                       '--sobol-repair', args.sobol_repair]+command[2:]
        return original(command, *positional, **keywords)

    subprocess.Popen = popen
    sys.path.insert(0, str(TOOLS))
    sys.argv = [str(TOOLS/'peptide_rgpe_merged_launch.py')]+remaining
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
