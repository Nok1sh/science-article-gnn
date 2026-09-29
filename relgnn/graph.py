"""Turning a RelBench database into a heterogeneous graph for the GNN."""

from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from relbench.base import Database
from relbench.modeling.utils import to_unix_time
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import Dataset as TorchFrameDataset
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index

from .text_embedder import GloveTextEmbedding

ColToStype = dict[str, dict[str, stype]]


@lru_cache(maxsize=1)
def text_embedder_cfg() -> TextEmbedderConfig:
    """The GloVe text embedder, loaded once per process."""
    return TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=torch.device("cpu")), batch_size=256
    )


def build_graph(
    db: Database,
    col_to_stype_dict: ColToStype,
    hidden_columns: list[tuple[str, str]],
    col_stats_dict: dict | None = None,
) -> tuple[HeteroData, dict]:
    """Build the primary-foreign key graph of ``db``, like relbench's
    ``make_pkey_fkey_graph``, returning the graph and the per-table column stats.

    Every table is a node type, every row a node, every foreign key a pair of edge
    types (``f2p_*`` and the reverse ``rev_f2p_*``).

    With ``col_stats_dict`` (the stats a model was trained with) the node features are
    encoded with those stats instead of ones computed from ``db``. Needed when ``db``
    differs from the training data (e.g. masked differently): otherwise categories are
    re-indexed and no longer match the embeddings the model learned.
    """
    hidden: dict[str, set[str]] = {}
    for table_name, col in hidden_columns:
        hidden.setdefault(table_name, set()).add(col)

    data = HeteroData()
    out_col_stats_dict = {}
    for table_name, table in db.table_dict.items():
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        # pkey, fkey and the task's hidden columns are not input features
        not_features = {table.pkey_col, *table.fkey_col_to_pkey_table}
        not_features |= hidden.get(table_name, set())
        col_to_stype = {
            col: stype_
            for col, stype_ in col_to_stype_dict[table_name].items()
            if col not in not_features
        }
        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            df = pd.DataFrame({"__const__": np.ones(len(df))})

        dataset = TorchFrameDataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg(),
        ).materialize(
            col_stats=None if col_stats_dict is None else col_stats_dict[table_name]
        )
        data[table_name].tf = dataset.tensor_frame
        out_col_stats_dict[table_name] = dataset.col_stats

        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = table.df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))[torch.from_numpy(mask.values)]
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges; "rev_" lets the PyG loader recognize reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()
    return data, out_col_stats_dict
