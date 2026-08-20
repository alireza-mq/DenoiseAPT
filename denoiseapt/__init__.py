"""DenoiseAPT research demonstration package.

The public surface is intentionally compact; detailed training components remain
available from their defining modules.
"""

__version__ = "0.2.2"

from .actions import SignalSession, apply_interval_action
from .concern import ConcernConfig, ConcernCues, compute_concern_cues
from .corruptions import CorruptionResult, apply_measurement_corruption

__all__ = [
    "ConcernConfig",
    "ConcernCues",
    "CorruptionResult",
    "SignalSession",
    "apply_interval_action",
    "apply_measurement_corruption",
    "compute_concern_cues",
]

# Heavy model/runtime modules are intentionally not imported here.  Importing
# ``denoiseapt.confirmation_data`` must remain a data-only operation, and all
# public runtime classes continue to be available from their defining modules
# (for example, ``denoiseapt.inference.DenoiseAPTPipeline``).
