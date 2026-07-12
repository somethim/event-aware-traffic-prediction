"""Pytest setup.

Point all data and output paths at a throwaway temp directory before any src module is
imported, so running the suite never reads or overwrites the real dataset under the repo.
src.config reads EATP_DATA_ROOT when it computes its path constants, so setting it here (before
the first src import) isolates the whole pipeline.
"""

import os
import tempfile

os.environ.setdefault("EATP_DATA_ROOT", tempfile.mkdtemp(prefix="eatp-test-"))
