import time

import pandas as pd

from data import load_data
from masks import Mask
from training import TrainConfig, train


def run_sweep(
    tasks: list[tuple[str, str]],
    masks: list[Mask],
    seeds: tuple[int, ...] = (42,),
    config: TrainConfig | None = None,
    csv_path: str | None = None,
    **overrides,
) -> pd.DataFrame:
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
                except ValueError:
                    continue
                record(
                    dataset, task, mask.name, seed, "original_trained",
                    metrics, time.time() - start,
                )

    return pd.DataFrame(rows)
