"""Masks: functions that replace the features of a RelBench database.

The graph is built from ``db.table_dict``: every table is a node type, every row a node,
every foreign key an edge, and the time column decides which rows a prediction may see.
A mask may change any *feature* column freely, but must keep, for every table:

* the number and order of rows (row i is node i),
* the primary key column (must stay 0..N-1),
* the foreign key columns (they are the edges); a foreign key may only be *removed*
  (set to missing), which deletes that edge,
* the time column (changing it breaks the no-future-leakage guarantee; only warned).

Column types (stypes) are taken from the *unmasked* data. A mask that changes what a
column holds (e.g. numbers into range labels) must update ``col_to_stype_dict`` too.
A model can only be evaluated on data whose column types match its training data.

Writing your own mask::

    def drop_names(db, col_to_stype_dict, rng):
        db.table_dict["drivers"].df["surname"] = "unknown"

    my_mask = Mask("drop_names", drop_names)
"""

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.base import Database, Table
from torch_frame import stype

ColToStype = dict[str, dict[str, stype]]
MaskFn = Callable[[Database, ColToStype, np.random.Generator], None]


@dataclass(frozen=True)
class Mask:
    """A named masking function ``fn(db, col_to_stype_dict, rng)`` that edits ``db``
    in place. ``name`` identifies the mask in logs and result tables."""

    name: str
    fn: MaskFn

    def __call__(
        self, db: Database, col_to_stype_dict: ColToStype, rng: np.random.Generator
    ) -> None:
        self.fn(db, col_to_stype_dict, rng)


def structural_columns(table: Table) -> list[str]:
    """Columns that define the graph and must not be masked."""
    cols = list(table.fkey_col_to_pkey_table)
    if table.pkey_col is not None:
        cols.append(table.pkey_col)
    if table.time_col is not None:
        cols.append(table.time_col)
    return cols


def feature_columns(table: Table) -> list[str]:
    """Columns that are free to mask."""
    protected = set(structural_columns(table))
    return [col for col in table.df.columns if col not in protected]


def noise(strength: float = 0.1) -> Mask:
    """Add Gaussian noise with std = ``strength`` * column std to every numerical
    feature. Integer codes typed as categorical (e.g. statusId) are left alone."""

    def fn(db: Database, col_to_stype_dict: ColToStype, rng: np.random.Generator):
        for table_name, table in db.table_dict.items():
            col_to_stype = col_to_stype_dict.get(table_name, {})
            df = table.df.copy()
            for col in feature_columns(table):
                if col_to_stype.get(col) != stype.numerical:
                    continue
                values = df[col].astype(float)
                std = values.std()
                if not np.isfinite(std) or std == 0:
                    continue
                df[col] = values + rng.normal(0.0, strength * std, size=len(df))
            table.df = df

    return Mask(f"noise({strength})", fn)


def shuffle(fraction: float = 0.5) -> Mask:
    """Shuffle a ``fraction`` of the rows of every feature column independently.

    Keeps each column's distribution but breaks the link between a row and its values.
    """

    def fn(db: Database, col_to_stype_dict: ColToStype, rng: np.random.Generator):
        for table in db.table_dict.values():
            df = table.df.copy()
            n = len(df)
            k = int(round(fraction * n))
            if k < 2:
                continue
            for col in feature_columns(table):
                idx = rng.choice(n, size=k, replace=False)
                values = df[col].to_numpy(copy=True)
                values[idx] = values[rng.permutation(idx)]
                df[col] = values
            table.df = df

    return Mask(f"shuffle({fraction})", fn)


def suppress(fraction: float = 0.5, tables: list[str] | None = None) -> Mask:
    """Replace a ``fraction`` of the values of every feature column with missing
    values (``fraction=1`` removes all feature values, leaving only graph structure and
    time). ``tables`` limits the mask to these tables."""

    def fn(db: Database, col_to_stype_dict: ColToStype, rng: np.random.Generator):
        for table_name, table in db.table_dict.items():
            if tables is not None and table_name not in tables:
                continue
            df = table.df.copy()
            for col in feature_columns(table):
                hit = rng.random(len(df)) < fraction
                if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_integer_dtype(
                    df[col]
                ):
                    df[col] = df[col].astype(object if df[col].dtype == bool else float)
                df.loc[hit, col] = None
            table.df = df

    return Mask(f"suppress({fraction})", fn)


def generalize(
    n_bins: int = 5,
    represent: str = "range",
    binning: str = "quantile",
    tables: list[str] | None = None,
) -> Mask:
    """Replace exact numbers with the range they fall in, e.g. 500 -> 400-600.

    Applies to numerical feature columns. Bins are ``n_bins`` quantiles (equally
    populated, ``binning="quantile"``) or equal-width intervals (``binning="width"``).

    ``represent`` decides what the model sees:

    * ``"range"``: the label "400-600"; the column becomes categorical, so the model
      only knows which range a value is in (no order, no distance between ranges),
    * ``"midpoint"``: the number 500; the column stays numerical, so a model trained
      on such data can also be evaluated on the original data and vice versa.
    """
    if represent not in ("range", "midpoint"):
        raise ValueError(f"represent must be 'range' or 'midpoint', got {represent!r}")
    if binning not in ("quantile", "width"):
        raise ValueError(f"binning must be 'quantile' or 'width', got {binning!r}")

    def fn(db: Database, col_to_stype_dict: ColToStype, rng: np.random.Generator):
        for table_name, table in db.table_dict.items():
            if tables is not None and table_name not in tables:
                continue
            col_to_stype = col_to_stype_dict.get(table_name, {})
            df = table.df.copy()
            for col in feature_columns(table):
                if col_to_stype.get(col) != stype.numerical:
                    continue
                values = df[col].astype(float)
                if values.nunique() <= 1:
                    continue
                if binning == "quantile":
                    bins = pd.qcut(values, n_bins, duplicates="drop")
                else:
                    bins = pd.cut(values, n_bins)
                # pandas widens the outer edges slightly; report the real extremes
                low, high = values.min(), values.max()
                if represent == "midpoint":
                    mids = bins.map(lambda b: b.mid, na_action="ignore")
                    df[col] = mids.astype(float)
                else:
                    labels = bins.map(
                        lambda b, lo=low, hi=high: f"{max(b.left, lo):g}-{min(b.right, hi):g}",
                        na_action="ignore",
                    )
                    df[col] = labels.astype(object)
                    col_to_stype[col] = stype.categorical
            table.df = df

    return Mask(f"generalize({n_bins},{represent},{binning})", fn)


def drop_links(fraction: float = 0.5, tables: list[str] | None = None) -> Mask:
    """Remove a ``fraction`` of the foreign key values (set them to missing), i.e.
    hide which row a record refers to; the corresponding graph edges disappear."""

    def fn(db: Database, col_to_stype_dict: ColToStype, rng: np.random.Generator):
        for table_name, table in db.table_dict.items():
            if tables is not None and table_name not in tables:
                continue
            df = table.df.copy()
            for col in table.fkey_col_to_pkey_table:
                hit = rng.random(len(df)) < fraction
                df[col] = df[col].astype(float)
                df.loc[hit, col] = np.nan
            table.df = df

    return Mask(f"drop_links({fraction})", fn)


def compose(*masks: Mask) -> Mask:
    """Apply several masks one after another, e.g.
    ``compose(generalize(5), suppress(0.3), drop_links(0.2))``."""

    def fn(db: Database, col_to_stype_dict: ColToStype, rng: np.random.Generator):
        for mask in masks:
            mask(db, col_to_stype_dict, rng)

    return Mask("+".join(mask.name for mask in masks), fn)


def from_dir(path: str) -> Mask:
    """Replace tables with ``<path>/<table>.parquet`` files, e.g. ones masked by an
    external tool starting from :func:`export_tables`. Tables without a file are kept."""
    folder = Path(path)

    def fn(db: Database, col_to_stype_dict: ColToStype, rng: np.random.Generator):
        files = sorted(folder.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {folder}")
        for file in files:
            if file.stem not in db.table_dict:
                raise KeyError(f"{file.name}: no table {file.stem!r} in the database")
            db.table_dict[file.stem].df = pd.read_parquet(file)

    return Mask(f"dir({folder.name})", fn)


def export_tables(db: Database, out_dir: str) -> None:
    """Write every table of ``db`` to ``<out_dir>/<table>.parquet``, plus
    ``schema.json`` listing the key and time columns that must be kept."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = {}
    for name, table in db.table_dict.items():
        table.df.to_parquet(out / f"{name}.parquet", index=False)
        schema[name] = {
            "pkey_col": table.pkey_col,
            "fkey_col_to_pkey_table": table.fkey_col_to_pkey_table,
            "time_col": table.time_col,
            "feature_columns": feature_columns(table),
        }
    (out / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def _same_values(a: pd.Series, b: pd.Series) -> bool:
    """Equal values and missing positions, ignoring the index and the exact dtype
    (e.g. int64 vs Int64 after a parquet round trip)."""
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if a.equals(b):
        return True
    missing = a.isna()
    if not missing.equals(b.isna()):
        return False
    return bool((a[~missing] == b[~missing]).all())


def _only_removed(before: pd.Series, after: pd.Series) -> bool:
    """Every value of ``after`` is either missing or equal to ``before``."""
    before = before.reset_index(drop=True)
    after = after.reset_index(drop=True)
    kept = after.notna()
    return bool((before[kept] == after[kept]).all())


def structure_snapshot(db: Database) -> dict[str, pd.DataFrame]:
    """Copy of the structural columns of every table, for :func:`validate_structure`."""
    return {
        name: table.df[structural_columns(table)].copy()
        for name, table in db.table_dict.items()
    }


def validate_structure(db: Database, reference: dict[str, pd.DataFrame]) -> None:
    """Check that masking kept the graph structure of every table intact."""
    for name, ref in reference.items():
        table = db.table_dict[name]
        df = table.df
        if len(df) != len(ref):
            raise ValueError(
                f"table {name!r}: {len(df)} rows after masking, expected {len(ref)}. "
                "Masking must not add, drop or reorder rows."
            )
        for col in ref.columns:
            if col not in df.columns:
                raise ValueError(f"table {name!r}: column {col!r} was removed")
            if _same_values(ref[col], df[col]):
                continue
            if col in table.fkey_col_to_pkey_table and _only_removed(ref[col], df[col]):
                continue  # links were removed, none were redirected
            if col == table.time_col:
                warnings.warn(
                    f"table {name!r}: time column {col!r} changed; temporal sampling "
                    "will no longer guarantee that predictions only see the past."
                )
            else:
                raise ValueError(
                    f"table {name!r}: key column {col!r} changed. Primary and "
                    "foreign keys define the graph and must stay as they are."
                )
