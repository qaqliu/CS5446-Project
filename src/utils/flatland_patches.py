"""Fix Flatland 4.3's tree-observation conflict prediction.

TreeObsForRailEnv compares a train's explored cells with the predicted
positions of other trains (`int_position in fast_delete(predicted_pos, h)`
and `fast_where(predicted_pos == int_position)`). In Flatland 4.3,
`coordinate_to_position` returns a Python list instead of a NumPy array, so
both tests compare a one-element list with integers and never match:
`dist_potential_conflict` stays infinite even when trains are predicted to
meet head-on. Restoring array semantics for these two names inside
flatland.envs.observations fixes it. They are used only in the predictor
branch there, so observations without a predictor are unchanged.
"""

import numpy as np
import flatland.envs.observations as _observations
from flatland.core.grid.grid_utils import coordinate_to_position as _coordinate_to_position


def _array_coordinate_to_position(depth, coords):
    return np.asarray(_coordinate_to_position(depth, coords))


def _array_delete(values, index):
    return np.delete(np.asarray(values), index)


if not getattr(_observations, "_cs5446_conflict_patch", False):
    _observations.coordinate_to_position = _array_coordinate_to_position
    _observations.fast_delete = _array_delete
    _observations._cs5446_conflict_patch = True
