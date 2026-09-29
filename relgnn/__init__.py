"""GNN experiments on RelBench entity tasks with masked data.

    from relgnn import load_data, masks, train

    original = load_data("rel-f1", "driver-position")
    masked = load_data("rel-f1", "driver-position", mask=masks.noise(0.1))

    model = train(original, epochs=25)
    model.evaluate(original)   # {"val": {...}, "test": {...}}
    model.evaluate(masked)     # the same model on masked data
"""

from . import masks
from .data import RelData, load_data
from .experiments import run_sweep
from .masks import Mask
from .training import TrainConfig, TrainedModel, evaluate, train

__all__ = [
    "Mask",
    "RelData",
    "TrainConfig",
    "TrainedModel",
    "evaluate",
    "load_data",
    "masks",
    "run_sweep",
    "train",
]
