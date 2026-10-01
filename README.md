# science-article-gnn

GNN on RelBench entity tasks, trained and evaluated on original and masked data.
Experiments are run from `gnn.ipynb` in Google Colab (GPU runtime).

## Files

| File | Responsibility |
| --- | --- |
| `gnn.ipynb` | entry point: setup, quick check, mask sweep, results summary |
| `data.py` | loads a RelBench task and its database, applies a mask, returns `RelData`; builds the graph on demand |
| `masks.py` | masks (`noise`, `shuffle`, `suppress`, `generalize`, `drop_links`, `compose`, `from_dir`) and the check that a mask kept rows and keys intact |
| `graph.py` | turns a database into a heterogeneous graph: tables → node types, rows → nodes, foreign keys → edges, columns → node features |
| `training.py` | `train` (training loop, best epoch by validation) and `TrainedModel` (`evaluate`, `predict`, `save`, `load`) |
| `experiments.py` | `run_sweep`: every task × mask × seed, both protocols, results as a table / CSV |
| `model.py` | the GNN: column encoders, temporal encoder, GraphSAGE / GAT layers, prediction head (from RelBench examples) |
| `text_embedder.py` | GloVe embeddings for text columns |
| `install.py` | installs the project and a `pyg-lib` build matching the installed torch / CUDA |
| `pyproject.toml` | dependencies |

## Usage

```python
import masks
from data import load_data
from training import train

original = load_data("rel-f1", "driver-position")
masked = load_data("rel-f1", "driver-position", mask=masks.suppress(0.5))

model = train(masked, epochs=15)
model.evaluate(masked)     # {"val": {"nmae": ...}, "test": {"nmae": ...}}
model.evaluate(original)
```

## Masks

A mask changes the feature values of the database; rows, primary keys, foreign key
targets and time columns stay as they are (checked when the data is loaded).

| Mask | Effect | Example |
| --- | --- | --- |
| `noise(strength)` | adds Gaussian noise with std = strength × column std to numerical columns | `64 → 71.3` |
| `shuffle(fraction)` | shuffles a fraction of the values within each feature column | values move between rows |
| `suppress(fraction)` | replaces a fraction of the feature values with missing values; `1.0` leaves only graph structure and time | `64 → NaN` |
| `generalize(n_bins, "range")` | replaces numbers with the range they fall in; the column becomes categorical | `64 → "56-70"` |
| `generalize(n_bins, "midpoint")` | replaces numbers with the middle of their range; the column stays numerical | `64 → 63.0` |
| `drop_links(fraction)` | removes a fraction of foreign keys, i.e. graph edges | `driverId 660 → NaN` |
| `compose(m1, m2, ...)` | applies several masks in turn | `compose(generalize(3), suppress(0.5))` |
| `from_dir(path)` | replaces tables with `<path>/<table>.parquet` masked by an external tool | |

`generalize` bins are quantiles (`binning="quantile"`, equally populated) or equal-width
intervals (`binning="width"`). Masks with `tables=[...]` (`suppress`, `generalize`,
`drop_links`) apply to the listed tables only.

A model can be evaluated only on data with the same column types as its training data:
a model trained on the original data cannot be evaluated on `generalize(..., "range")`
data, and vice versa.

Own mask: a function editing the database in place, wrapped in `Mask`:

```python
from masks import Mask

def drop_names(db, col_to_stype_dict, rng):
    db.table_dict["drivers"].df["surname"] = "unknown"

data = load_data("rel-f1", "driver-position", mask=Mask("drop_names", drop_names))
```

## Sweep

`run_sweep` trains and evaluates every task × mask × seed combination and returns a
table (also written to `csv_path` after every run). Two protocols per mask:

- `masked_trained`: trained and evaluated on the masked data,
- `original_trained`: trained on the original data, evaluated on the masked data.

```python
from experiments import run_sweep

df = run_sweep(
    [("rel-f1", "driver-position"), ("rel-f1", "driver-dnf")],
    [masks.noise(1.0), masks.suppress(0.5)],
    seeds=(42, 43),
    csv_path="sweep.csv",
    epochs=15,
)
```
