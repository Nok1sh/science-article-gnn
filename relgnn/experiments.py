"""Sweeps over tasks, masks and seeds."""

import time

import pandas as pd

from .data import load_data
from .masks import Mask
from .training import TrainConfig, train


def run_sweep(
    tasks: list[tuple[str, str]],
    masks: list[Mask],
    seeds: tuple[int, ...] = (42,),
    config: TrainConfig | None = None,
    csv_path: str | None = None,
    **overrides,
) -> pd.DataFrame:
    """Train and evaluate every (task, mask, seed) combination.

    For each task and seed a reference model is trained on the original data. Then for
    every mask two numbers are recorded:

    * ``masked_trained``: a model trained on the masked data, evaluated on it (the
      realistic case: only the masked database is available),
    * ``original_trained``: the reference model evaluated on the masked data (how much
      the model relies on the exact values). Empty when the mask changes column types.

    One row per (task, mask, seed, protocol, split) with the task's metrics as columns.
    ``csv_path`` saves the table after every run, so partial results survive a crash.
    """
    config = TrainConfig(verbose=False) if config is None else config
    rows: list[dict] = []

    def record(dataset, task, mask_name, seed, protocol, metrics, seconds):
        for split, values in metrics.items():
            rows.append(
                {
                    "task": f"{dataset}/{task}",
                    "mask": mask_name,
                    "seed": seed,
                    "protocol": protocol,
                    "split": split,
                    **values,
                    "seconds": round(seconds, 1),
                }
            )
        if csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    for dataset, task in tasks:
        original = load_data(dataset, task)
        for seed in seeds:
            start = time.time()
            reference = train(original, config, seed=seed, **overrides)
            record(
                dataset, task, "original", seed, "original_trained",
                reference.evaluate(original), time.time() - start,
            )
            print(f"{original.name} seed={seed}: {rows[-1]}")

            for mask in masks:
                masked = load_data(dataset, task, mask=mask, seed=seed)

                start = time.time()
                model = train(masked, config, seed=seed, **overrides)
                record(
                    dataset, task, mask.name, seed, "masked_trained",
                    model.evaluate(masked), time.time() - start,
                )
                print(f"{masked.name} seed={seed} masked_trained: {rows[-1]}")

                start = time.time()
                try:
                    metrics = reference.evaluate(masked)
                except ValueError:  # the mask changed column types
                    continue
                record(
                    dataset, task, mask.name, seed, "original_trained",
                    metrics, time.time() - start,
                )

    return pd.DataFrame(rows)
