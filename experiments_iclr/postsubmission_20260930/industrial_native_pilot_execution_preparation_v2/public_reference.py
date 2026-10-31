"""Disposable public-only oracle. Targets/all-label provider APIs are absent."""
from __future__ import annotations
from pathlib import Path
from typing import Literal,cast
import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from types import SimpleNamespace
from native_transforms import Setting,NumPolicy,CatPolicy,impute_nans,transform_num_features,transform_cat_features
GraphData = dict
GraphLandInfo = dict
PartKey = str
DATA_PARTS = ("train","val","test")
constants = SimpleNamespace(GRAPHLAND_DATASETS=("tolokers-2",))

def load_dict(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as file:
        d = yaml.safe_load(file)
    return d


def reference_public_load(
    path: str | Path,
    *,
    internal_split_name: Literal["RL", "RH", "TH"],
    **kwargs,
) -> GraphData:
    del kwargs
    path = Path(path).resolve()
    name = path.name

    if name not in constants.GRAPHLAND_DATASETS:
        raise ValueError(f"Unknown dataset {name=}!")

    # >>> Load info
    info = cast(GraphLandInfo, load_dict(path / "info.yaml"))

    # >>> Load features
    features_df = pd.read_csv(path / "features.csv", index_col=0).astype(np.float32)

    # >>> Drop constant features
    features_df = features_df.loc[:, features_df.apply(pd.Series.nunique) != 1]
    columns_remained = list(features_df.columns)

    # Public-only adaptation: node count comes from frozen public feature rows.
    public_node_count = len(features_df)

    # >>> Load & prepare data split
    masks_df = pd.read_csv(
        path / f"split_masks_{internal_split_name}.csv",
        index_col=0,
    ).astype(bool)
    masks: dict[PartKey, np.ndarray] = {
        part: masks_df[part].values for part in DATA_PARTS
    }  # type: ignore

    # >> Separate features of different types
    # NOTE: fraction features in GraphLand are treated as numerical features
    frac_features_names = [
        feature_name
        for feature_name in info["fraction_features_names"]
        if feature_name in columns_remained
    ]
    frac_features = (
        features_df.loc[:, frac_features_names].values.astype(np.float32)
        if frac_features_names
        else None
    )
    if frac_features is not None:
        assert not np.isnan(frac_features).any().item(), (
            "Fraction features can not contain nans"
        )

    # NOTE: categorical features in GraphLand do not contain nans
    # so casting to integer dtype should not throw exceptions
    cat_features_names = [
        feature_name
        for feature_name in info["categorical_features_names"]
        if feature_name in columns_remained
    ]
    cat_features = (
        features_df.loc[:, cat_features_names].values.astype(np.int32)
        if cat_features_names
        else None
    )

    num_features_names = [
        feature_name
        for feature_name in info["numerical_features_names"]
        if feature_name not in frac_features_names and feature_name in columns_remained
    ]
    num_features = (
        features_df.loc[:, num_features_names].values.astype(np.float32)
        if num_features_names
        else None
    )

    # >>> Construct graph
    edges_df = pd.read_csv(path / "edgelist.csv")
    edges = torch.from_numpy(edges_df.values[:, :2]).T
    graph = dgl.graph(
        data=(edges[0], edges[1]),
        num_nodes=public_node_count,
        idtype=torch.int32,
    )

    graph_data = {
        "name": name,
        "graph": graph,
        "masks": masks,
        "num_features": num_features,
        "cat_features": cat_features,
        "frac_features": frac_features,
    }
    return cast(GraphData, graph_data)


def postprocess_reference(data):
    cats=data["cat_features"]
    if cats is not None:
        mask=np.array([len(np.unique(x))>1 for x in cats.T])
        assert mask.any().item()
        cats=cats[:,mask];data["cat_features"]=cats
    fractions=data["frac_features"]
    if fractions is not None:
        mask=np.array([np.all((x==0.)|(x==1.)) for x in fractions.T])
        if mask.any().item():
            binary=fractions[:,mask].astype(np.float32)
            cats=data["cat_features"]
            data["cat_features"]=np.concatenate([cats,binary],axis=1) if cats is not None else binary
            data["frac_features"]=None if mask.all().item() else fractions[:,~mask]
    graph=dgl.remove_self_loop(data["graph"])
    graph=dgl.to_simple(graph)
    data["graph"]=dgl.to_bidirected(graph)
    return data

def reference_transform(data,policies):
    # Native graph-feature transform order is numerical,fraction,categorical.
    for key in ("num_features","frac_features","cat_features"):
        if data[key] is None:continue
        policy=policies[{"num_features":"num_policy","frac_features":"frac_policy","cat_features":"cat_policy"}[key]]
        fn=transform_cat_features if key=="cat_features" else transform_num_features
        kwargs={"setting":Setting.TRANSDUCTIVE,"policy":policy}
        if key!="cat_features":kwargs["seed"]=policies["seed"]
        data[key]=fn(data[key],data["masks"],**kwargs)
    return data
