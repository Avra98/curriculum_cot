"""Recovered wrapper around the surviving compiled latent SFT module."""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from _sourceless_loader import load_pyc_into_globals


_MODULE = load_pyc_into_globals(__file__, "sft_latent_multi_output_train.cpython-311.pyc", globals())


if __name__ == "__main__" and hasattr(_MODULE, "main"):
    _MODULE.main()
