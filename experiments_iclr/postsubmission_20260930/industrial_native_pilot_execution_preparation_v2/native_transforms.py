"""Exact native feature transforms; typed inputs supplied by closed public loader."""
from __future__ import annotations
import enum
import numpy as np
import sklearn.impute
import sklearn.preprocessing
PartKey = str

class Setting(enum.Enum):
    TRANSDUCTIVE = "transductive"
    INDUCTIVE = "inductive"


class NumPolicy(enum.Enum):
    STANDARD = "standard"
    QUANTILE_NORMAL = "quantile-normal"
    QUANTILE_UNIFORM = "quantile-uniform"
    NONE = "none"


def impute_nans(
    features: np.ndarray,
    masks: dict[PartKey, np.ndarray],
    setting: str | Setting,
    strategy: str = "most_frequent",
) -> np.ndarray:
    setting = Setting(setting)
    assert setting == Setting.TRANSDUCTIVE

    mask_seen = (
        np.ones_like(masks["train"], dtype=bool)
        if setting == Setting.TRANSDUCTIVE
        else masks["train"]
        # NOTE: this is not correct for inductive setting, as train graph
        # can have unlabeled nodes that must be taken into account
    )

    imputer = sklearn.impute.SimpleImputer(strategy=strategy)
    imputer.fit(features[mask_seen])
    features = imputer.transform(features)  # type: ignore

    return features


def transform_num_features(
    features: np.ndarray,
    masks: dict[PartKey, np.ndarray],
    *,
    setting: str | Setting,
    policy: None | str | NumPolicy,
    seed: None | int,
) -> np.ndarray:
    if policy is None:
        features = impute_nans(features, masks, setting)
        return features

    policy = NumPolicy(policy)
    # NOTE: throws exception in case of unknown value

    setting = Setting(setting)
    mask_seen = (
        np.ones_like(masks["train"], dtype=bool)
        if setting == Setting.TRANSDUCTIVE
        else masks["train"]
        # NOTE: this is not correct for inductive setting, as train graph
        # can have unlabeled nodes that must be taken into account
    )

    # NOTE: reserve a separate variable to avoid modifications in the original features
    features_seen = features[mask_seen]

    if policy == NumPolicy.STANDARD:
        normalizer = sklearn.preprocessing.StandardScaler()

    elif policy in [NumPolicy.QUANTILE_NORMAL, NumPolicy.QUANTILE_UNIFORM]:
        distribution = {
            NumPolicy.QUANTILE_NORMAL: "normal",
            NumPolicy.QUANTILE_UNIFORM: "uniform",
        }[policy]

        assert seed is not None
        normalizer = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(features_seen.shape[0] // 30, 1000), 10),
            output_distribution=distribution,  # type: ignore
            subsample=1_000_000_000,
            random_state=seed,
        )
        features_seen = features_seen + np.random.RandomState(seed).normal(
            0.0, 1e-5, features_seen.shape
        ).astype(features_seen.dtype)
    elif policy == NumPolicy.NONE:
        normalizer = sklearn.preprocessing.FunctionTransformer(func=None)
    else:
        raise ValueError(f"Unknown {policy=}")

    normalizer.fit(features_seen)
    features_trasformed = normalizer.transform(features)
    features_trasformed = impute_nans(features_trasformed, masks, setting)  # type: ignore

    return features_trasformed  # type: ignore


class CatPolicy(enum.Enum):
    ORDINAL = "ordinal"
    ONE_HOT = "one-hot"


def transform_cat_features(
    features: np.ndarray,
    masks: dict[PartKey, np.ndarray],
    *,
    setting: str | Setting,
    policy: None | str | CatPolicy,
) -> np.ndarray:
    if policy is None:
        return features

    # NOTE: throws exception in case of unknown value
    policy = CatPolicy(policy)

    setting = Setting(setting)
    mask_seen = (
        np.ones_like(masks["train"], dtype=bool)
        if setting == Setting.TRANSDUCTIVE
        else masks["train"]
        # NOTE: this is not correct for inductive setting, as train graph
        # can have unlabeled nodes that must be taken into account
    )

    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int32,
    ).fit(features[mask_seen])
    features_encoded = encoder.transform(features)

    if policy == CatPolicy.ORDINAL:
        return features_encoded

    if policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            drop="if_binary",
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.float32,
        )

    encoder.fit(features_encoded[mask_seen])
    features_transformed = encoder.transform(features_encoded)
    return features_transformed  # type: ignore

