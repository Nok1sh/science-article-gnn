"""Training a GNN on :class:`RelData` and evaluating it on any compatible data."""

import copy
import math
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from relbench.base import TaskType
from relbench.modeling.graph import get_node_train_table_input
from relbench.submit import evaluate_task, write_prediction_table
from torch.nn import BCEWithLogitsLoss, L1Loss, Module
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from .data import RelData
from .graph import ColToStype
from .model import Model


@dataclass
class TrainConfig:
    lr: float = 0.005
    epochs: int = 10
    batch_size: int = 512
    channels: int = 128
    aggr: str = "sum"
    num_layers: int = 2
    num_neighbors: int = 128
    gnn: str = "sage"  # "sage" or "gat"
    gat_heads: int = 4
    gat_dropout: float = 0.0
    temporal_strategy: str = "uniform"  # "uniform" or "last"
    max_steps_per_epoch: int = 2000
    num_workers: int = 0
    seed: int = 42
    device: str | None = None  # None: cuda if available
    verbose: bool = True


def _device(config: TrainConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _objective(data: RelData) -> tuple[Module, str, bool]:
    """Loss function, metric to select the best epoch by, and whether higher is better."""
    task_type = data.task.task_type
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return BCEWithLogitsLoss(), "roc_auc", True
    if task_type == TaskType.REGRESSION:
        return L1Loss(), "nmae", False
    raise ValueError(f"Unsupported task type: {task_type}")


def _regression_clamp(data: RelData) -> tuple[float, float] | None:
    """2nd/98th percentiles of the train target, used to clip regression predictions."""
    if data.task.task_type != TaskType.REGRESSION:
        return None
    target = data.table("train").df[data.task.target_col].to_numpy()
    low, high = np.percentile(target, [2, 98])
    return float(low), float(high)


def _build_model(graph: HeteroData, col_stats_dict: dict, config: TrainConfig) -> Model:
    return Model(
        data=graph,
        col_stats_dict=col_stats_dict,
        num_layers=config.num_layers,
        channels=config.channels,
        out_channels=1,
        aggr=config.aggr,
        norm="batch_norm",
        gnn=config.gnn,
        gat_heads=config.gat_heads,
        gat_dropout=config.gat_dropout,
    )


def _loader(
    graph: HeteroData, data: RelData, split: str, config: TrainConfig, shuffle: bool
) -> NeighborLoader:
    table_input = get_node_train_table_input(table=data.table(split), task=data.task)
    return NeighborLoader(
        graph,
        num_neighbors=[
            int(config.num_neighbors / 2**i) for i in range(config.num_layers)
        ],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=config.batch_size,
        temporal_strategy=config.temporal_strategy,
        shuffle=shuffle,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )


@dataclass
class TrainedModel:
    """A trained GNN plus everything needed to apply it to other data.

    ``col_to_stype_dict`` and ``col_stats_dict`` describe how the model saw its
    training data; any data it is evaluated on is encoded the same way.
    """

    model: Model
    config: TrainConfig
    dataset_name: str
    task_name: str
    include_task_tables: str
    col_to_stype_dict: ColToStype
    col_stats_dict: dict
    clamp: tuple[float, float] | None
    trained_on: str
    best_epoch: int
    history: list[dict] = field(default_factory=list)

    @property
    def best_val_metrics(self) -> dict:
        return self.history[self.best_epoch - 1]["val"] if self.history else {}

    def _check_compatible(self, data: RelData) -> None:
        mine = (self.dataset_name, self.task_name, self.include_task_tables)
        theirs = (data.dataset_name, data.task_name, data.include_task_tables)
        if mine != theirs:
            raise ValueError(
                f"model trained on {mine} cannot be evaluated on {theirs}: dataset, "
                "task and include_task_tables must match"
            )
        changed = [
            f"{table}.{col}: {stype_.value} -> {data.col_to_stype_dict[table][col].value}"
            for table, cols in self.col_to_stype_dict.items()
            for col, stype_ in cols.items()
            if data.col_to_stype_dict.get(table, {}).get(col, stype_) != stype_
        ]
        if changed:
            raise ValueError(
                f"model trained on {self.trained_on} cannot be evaluated on "
                f"{data.name}: column types differ ({', '.join(changed)}). "
                "E.g. use generalize(represent='midpoint') to keep numbers numerical."
            )

    @torch.no_grad()
    def predict(self, data: RelData, split: str) -> np.ndarray:
        """Predictions for the rows of ``data.table(split)``, in the same order."""
        self._check_compatible(data)
        graph, _ = data.graph(self.col_stats_dict)
        device = next(self.model.parameters()).device
        loader = _loader(graph, data, split, self.config, shuffle=False)

        self.model.eval()
        pred_list = []
        for batch in tqdm(loader, disable=not self.config.verbose, desc=split):
            batch = batch.to(device)
            pred = self.model(batch, data.task.entity_table)
            if self.clamp is not None:
                pred = torch.clamp(pred, *self.clamp)
            if data.task.task_type == TaskType.BINARY_CLASSIFICATION:
                pred = torch.sigmoid(pred)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
        return torch.cat(pred_list, dim=0).numpy()

    def evaluate(
        self, data: RelData, splits: tuple[str, ...] = ("val", "test")
    ) -> dict[str, dict[str, float]]:
        """Metrics per split, e.g. ``{"val": {"nmae": 0.44}, "test": {"nmae": 0.60}}``.

        Test labels are hidden: test predictions are written to a temporary CSV and
        scored by relbench against the hosted labels.
        """
        results = {}
        for split in splits:
            pred = self.predict(data, split)
            if split == "test":
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "pred.csv"
                    write_prediction_table(data.task, pred, path)
                    results[split] = evaluate_task(
                        f"{data.dataset_name}/{data.task_name}", path
                    )
            else:
                results[split] = data.task.evaluate(pred, data.table(split))
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "dataset_name": self.dataset_name,
                "task_name": self.task_name,
                "include_task_tables": self.include_task_tables,
                "col_to_stype_dict": self.col_to_stype_dict,
                "col_stats_dict": self.col_stats_dict,
                "clamp": self.clamp,
                "trained_on": self.trained_on,
                "best_epoch": self.best_epoch,
                "history": self.history,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, data: RelData) -> "TrainedModel":
        """Load a model saved with :meth:`save`; ``data`` is any compatible data, used
        only to build the model's layers."""
        # weights_only=False: the file also stores column stats and stypes, not just
        # tensors. Only load files you created yourself.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = TrainConfig(**ckpt["config"])
        trained = cls(
            model=None,  # type: ignore[arg-type]  # built below, needs the graph
            config=config,
            dataset_name=ckpt["dataset_name"],
            task_name=ckpt["task_name"],
            include_task_tables=ckpt["include_task_tables"],
            col_to_stype_dict=ckpt["col_to_stype_dict"],
            col_stats_dict=ckpt["col_stats_dict"],
            clamp=ckpt["clamp"],
            trained_on=ckpt["trained_on"],
            best_epoch=ckpt["best_epoch"],
            history=ckpt["history"],
        )
        trained._check_compatible(data)
        graph, _ = data.graph(trained.col_stats_dict)
        model = _build_model(graph, trained.col_stats_dict, config)
        model.load_state_dict(ckpt["state_dict"])
        trained.model = model.to(_device(config))
        return trained


def train(data: RelData, config: TrainConfig | None = None, **overrides) -> TrainedModel:
    """Train a GNN on ``data`` and return the weights of the best validation epoch.

    Hyperparameters come from ``config`` (defaults if omitted); keyword arguments
    override single fields, e.g. ``train(data, epochs=25, lr=0.001)``.
    """
    config = replace(config or TrainConfig(), **overrides)
    seed_everything(config.seed)
    device = _device(config)

    graph, col_stats_dict = data.graph()
    loss_fn, tune_metric, higher_is_better = _objective(data)
    train_loader = _loader(graph, data, "train", config, shuffle=True)

    trained = TrainedModel(
        model=_build_model(graph, col_stats_dict, config).to(device),
        config=config,
        dataset_name=data.dataset_name,
        task_name=data.task_name,
        include_task_tables=data.include_task_tables,
        col_to_stype_dict=data.col_to_stype_dict,
        col_stats_dict=col_stats_dict,
        clamp=_regression_clamp(data),
        trained_on=data.name,
        best_epoch=0,
    )
    model = trained.model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    entity_table = data.task.entity_table

    best_state = None
    best_val_metric = -math.inf if higher_is_better else math.inf
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_accum = count_accum = 0
        total_steps = min(len(train_loader), config.max_steps_per_epoch)
        for step, batch in enumerate(
            tqdm(train_loader, total=total_steps, disable=not config.verbose)
        ):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch, entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            loss = loss_fn(pred.float(), batch[entity_table].y.float())
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)
            if step + 1 >= config.max_steps_per_epoch:
                break

        val_metrics = trained.evaluate(data, splits=("val",))["val"]
        trained.history.append(
            {"epoch": epoch, "train_loss": loss_accum / count_accum, "val": val_metrics}
        )
        if config.verbose:
            print(
                f"Epoch: {epoch:02d}, Train loss: {loss_accum / count_accum:.4f}, "
                f"Val metrics: {val_metrics}"
            )

        metric = val_metrics[tune_metric]
        if (higher_is_better and metric >= best_val_metric) or (
            not higher_is_better and metric <= best_val_metric
        ):
            best_val_metric = metric
            trained.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return trained


def evaluate(
    model: TrainedModel, data: RelData, splits: tuple[str, ...] = ("val", "test")
) -> dict[str, dict[str, float]]:
    """Same as ``model.evaluate(data, splits)``."""
    return model.evaluate(data, splits)
