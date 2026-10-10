#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run every Python analysis package in this repository and self-check it.

    python run_all.py

Each package is executed in its own directory so that it resolves its own
``./data`` and ``./outputs`` folders.  The exit code is non-zero if any
package fails to run or if any regenerated value disagrees with the number
printed in the Supplementary Information.

Only the *verification* entry points are run here.  The figure-drawing scripts
are deliberately not invoked, because they write into tracked directories; run
them by hand, as described in each package README.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# label, folder, script, extra arguments
PACKAGES = [
    ("Supplementary Note 2 / Fig. 5", "Note2_effective_rank_memory_capacity",
     "analyze_effective_rank_mc.py", []),
    ("Supplementary Note 6 / Figs. 10-14", "Note6_repeatability_stability_noise",
     "analyze_note6.py", ["--no-figures"]),
    ("Raw-data provenance (Note 6)", "Note6_repeatability_stability_noise",
     "check_raw_provenance.py", []),
    ("Supplementary Fig. 15", "FigS15_capacitance", "check_tau_values.py", []),
]


def main() -> int:
    print("Python %s" % sys.version.split()[0])
    print("repository root: %s" % HERE)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    results = []
    for label, folder, script, extra in PACKAGES:
        path = HERE / folder / script
        print()
        print("#" * 96)
        print("# %s" % label)
        print("# %s" % path.relative_to(HERE))
        print("#" * 96)
        if not path.is_file():
            print("MISSING: %s" % path)
            results.append((label, "MISSING", None))
            continue
        proc = subprocess.run([sys.executable, script] + extra, cwd=str(path.parent), env=env)
        results.append((label, "exit %d" % proc.returncode, proc.returncode))

    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    ok = True
    for label, status, code in results:
        good = code == 0
        ok &= good
        print("%-42s %-12s %s" % (label, status, "OK" if good else "FAILED"))
    print("-" * 96)
    print("overall: %s" % ("ALL PACKAGES OK" if ok else "SOME PACKAGES FAILED"))
    print()
    print("Not run by this script:")
    print("  Note5_empirical_model   MATLAB commands and the two figure scripts")
    print("  FigS15_capacitance      the figure script")
    print("See the package READMEs; none of them writes into a tracked directory")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
