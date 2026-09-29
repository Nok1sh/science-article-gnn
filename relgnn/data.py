"""Loading a RelBench entity task as (optionally masked) data ready for the GNN."""

import copy
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from relbench.base import AutoCompleteTask, Database, Dataset, EntityTask, Table
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
from torch_geometric.data import HeteroData

from relbench import load_dataset

from .graph import ColToStype, build_graph
from .masks import Mask, structure_snapshot, validate_structure


@lru_cache(maxsize=None)
def _load_dataset(dataset_name: str) -> Dataset:
    return load_dataset(dataset_name)


@lru_cache(maxsize=None)
def _load_task(dataset_name: str, task_name: str) -> EntityTask:
    task = _load_dataset(dataset_name).load_task(task_name)
    if not isinstance(task, EntityTask):
        raise ValueError(f"{dataset_name}/{task_name} is not an entity task")
    return task


@lru_cache(maxsize=None)
def _base_db(dataset_name: str, full: bool) -> tuple[Database, ColToStype]:
    """The unmasked database and its column types, computed once per dataset.

    ``full=False`` cuts the rows after the test timestamp (forecasting tasks);
    autocomplete tasks need them (``full=True``): the rows to complete are there.
    Callers must not modify the result; :func:`load_data` works on a copy.
    """
    db = _load_dataset(dataset_name).get_db(upto_test_timestamp=not full)
    # get_stype_proposal samples rows with numpy's global RNG; fix it so the proposed
    # types never depend on what ran before
    state = np.random.get_state()
    np.random.seed(0)
    try:
        col_to_stype_dict = get_stype_proposal(db)
    finally:
        np.random.set_state(state)
    return db, col_to_stype_dict


def _copy_db(db: Database) -> Database:
    return Database(
        {
            name: Table(
                df=table.df.copy(),
                fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )
            for name, table in db.table_dict.items()
        }
    )


def _add_task_label_tables(
    db: Database, dataset: Dataset, task_name: str, mode: str
) -> ColToStype:
    """Add (time-censored) label tables to ``db`` as extra features, in place.

    ``mode``: 'none', 'current_only' (labels of ``task_name``) or 'all' (labels of every
    entity task of the dataset). Returns the stypes of the added tables.
    """
    if mode == "all":
        task_names = dataset.get_task_names()
    elif mode == "current_only":
        task_names = [task_name]
    elif mode == "none":
        task_names = []
    else:
        raise ValueError(f"Unknown include_task_tables={mode!r}")

    added: ColToStype = {}
    for name in task_names:
        t = dataset.load_task(name)
        if not isinstance(t, EntityTask):
            continue
        label_df = pd.concat(
            [
                t.get_table("train").df,
                t.get_table("val").df,
                # test set not included b/c labels are not revealed
            ]
        )
        # time-censoring labels: a label becomes known only after its window closes,
        # so shift its time by timedelta (i.e. no leakage)
        label_df[t.time_col] = label_df[t.time_col] + t.timedelta
        table_name = f"{name}_labels"
        db.table_dict[table_name] = Table(
            df=label_df,
            fkey_col_to_pkey_table={t.entity_col: t.entity_table},
            pkey_col=None,
            time_col=t.time_col,
        )
        added[table_name] = {
            t.entity_col: stype.numerical,
            t.time_col: stype.timestamp,
            t.target_col: stype.numerical,
        }
    return added


@dataclass
class RelData:
    """A RelBench entity task together with its (possibly masked) database.

    Create it with :func:`load_data`. The graph is built lazily: once with the data's
    own column stats for training, and once per foreign model's stats for evaluation.
    """

    dataset_name: str
    task_name: str
    task: EntityTask
    db: Database
    col_to_stype_dict: ColToStype
    mask: Mask | None
    include_task_tables: str
    _graphs: dict[int, tuple[dict, HeteroData]] = field(
        default_factory=dict, repr=False
    )
    _own_col_stats: dict | None = field(default=None, repr=False)
    _tables: dict[str, Table] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        mask = self.mask.name if self.mask is not None else "original"
        return f"{self.dataset_name}/{self.task_name} [{mask}]"

    def table(self, split: str) -> Table:
        """The task's train / val / test table (test comes without the target)."""
        if split not in self._tables:
            self._tables[split] = self.task.get_table(split)
        return self._tables[split]

    def graph(self, col_stats_dict: dict | None = None) -> tuple[HeteroData, dict]:
        """The graph of this data and the column stats its features are encoded with.

        ``col_stats_dict=None`` uses the data's own stats (for training); passing a
        model's stats encodes this data the way that model saw its training data.
        """
        if col_stats_dict is None:
            if self._own_col_stats is None:
                data, stats = build_graph(
                    self.db, self.col_to_stype_dict, self.task.hidden_columns()
                )
                self._own_col_stats = stats
                self._graphs[id(stats)] = (stats, data)
            col_stats_dict = self._own_col_stats

        key = id(col_stats_dict)
        if key not in self._graphs:
            data, _ = build_graph(
                self.db,
                self.col_to_stype_dict,
                self.task.hidden_columns(),
                col_stats_dict,
            )
            # the stats object is kept alive by the cache, so its id cannot be reused
            self._graphs[key] = (col_stats_dict, data)
        return self._graphs[key][1], col_stats_dict


def load_data(
    dataset: str,
    task: str,
    mask: Mask | None = None,
    include_task_tables: str = "none",
    seed: int = 0,
) -> RelData:
    """Load a RelBench entity task, masking its database with ``mask`` if given.

    Args:
        dataset: e.g. ``"rel-f1"``.
        task: e.g. ``"driver-position"``.
        mask: e.g. ``masks.noise(0.1)``; ``None`` keeps the original data.
        include_task_tables: 'none', 'current_only' or 'all': add past labels as
            (time-censored) features.
        seed: seed of the random generator passed to the mask.

    Column types are proposed on the unmasked database, so the original and every
    masked version of a dataset share them, unless a mask changes a column's type.
    """
    task_obj = _load_task(dataset, task)
    base_db, base_col_to_stype_dict = _base_db(
        dataset, full=isinstance(task_obj, AutoCompleteTask)
    )
    db = _copy_db(base_db)
    col_to_stype_dict = copy.deepcopy(base_col_to_stype_dict)

    if mask is not None:
        reference = structure_snapshot(db)
        mask(db, col_to_stype_dict, np.random.default_rng(seed))
        validate_structure(db, reference)

    col_to_stype_dict.update(
        _add_task_label_tables(db, _load_dataset(dataset), task, include_task_tables)
    )
    return RelData(
        dataset_name=dataset,
        task_name=task,
        task=task_obj,
        db=db,
        col_to_stype_dict=col_to_stype_dict,
        mask=mask,
        include_task_tables=include_task_tables,
    )
