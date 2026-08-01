"""GeneaKit: a set of functions for pedigree analysis.

The public interface is re-exported from the modules below, so that a user
writes `geneakit.phi(...)` rather than `geneakit.compute.phi(...)`. The paths
to the bundled datasets are built first, which is why the imports do not all
sit at the top of the file.
"""

import os
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('geneakit')
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = 'unknown'

path = os.path.dirname(os.path.realpath(__file__)) + "/datasets/"
genea140 = path + "genea140.csv"
geneaJi = path + "geneaJi.csv"
pop140 = path + "pop140.csv"

from .create import *      # noqa: E402,F401,F403
from .output import *      # noqa: E402,F401,F403
from .identify import *    # noqa: E402,F401,F403
from .extract import *     # noqa: E402,F401,F403
from .describe import *    # noqa: E402,F401,F403
from .compute import *     # noqa: E402,F401,F403
from .graph import *       # noqa: E402,F401,F403
