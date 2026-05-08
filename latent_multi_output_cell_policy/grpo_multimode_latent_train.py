from __future__ import annotations

"""GRPO entrypoint with latent-mode dispatch aligned to latent SFT.

This script intentionally reuses ``grpo_residual_projector_latent_train`` after
the underlying module was updated to route GRPO forward/generation through the
selected ``--latent_mode``. Keeping this thin entrypoint makes the experiment
name explicit while preserving the existing CLI and checkpoint format.

Supported modes match ``sft_latent_multi_output_train.py``:

* ``residual``
* ``fixed_slots``
* ``recurrent_hidden``
* ``latent_seeds``
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from latent_multi_output_cell_policy.grpo_residual_projector_latent_train import main


if __name__ == "__main__":
    main()
