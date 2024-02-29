from pathlib import Path
from dataclasses import dataclass
from src.attribute import AttrType

import pandas as pd
import numpy as np


@dataclass
class Dataset:
    ds_dir: Path = Path.cwd() / "datasets"

    @staticmethod
    def load_dataset(ds_name: str, label_name: str) -> tuple[np.ndarray, np.ndarray]:
        filename = ds_name + ".parquet"

        if not (Dataset.ds_dir / filename).exists():
            raise FileNotFoundError(f"The {filename} does not exists.")

        df = pd.read_parquet((Dataset.ds_dir / filename).absolute())
        structure = Dataset._attr_structure(df.loc[:, df.columns != label_name])
        return df.to_numpy(), structure

    @staticmethod
    def _attr_structure(data: pd.DataFrame) -> np.ndarray:
        types = np.repeat("", data.shape[1])
        c_attr = set(data._get_numeric_data().columns)  # type: ignore
        d_attr = set(data.columns) - c_attr

        cidx = data.columns.get_indexer(c_attr)
        didx = data.columns.get_indexer(d_attr)

        types[cidx] = AttrType.C.value
        types[didx] = AttrType.D.value
        return types

    @staticmethod
    def dataset_domain(data: np.ndarray) -> dict[int, list]:
        domain = {}
        for col in range(data.shape[1]):
            domain[col] = list(set(data[:, col]))

        return domain

    @staticmethod
    def discretize_dataset(data: np.ndarray, data_struc: np.ndarray, nbins=20):
        for i, a in enumerate(data_struc):
            if a == "c":
                att_d = data[:, i]
                buckets = np.linspace(np.min(att_d), np.max(att_d), num=nbins + 1)
                bucketized = np.searchsorted(buckets, att_d)
                data[:, i] = bucketized
                data_struc[i] = "d"

        return data, data_struc

    @staticmethod
    def only_discrete(data: np.ndarray, data_struc: np.ndarray):
        disc_mask = data_struc == AttrType.D.value
        return data[:, np.hstack([disc_mask, True])], data_struc[disc_mask]

    @staticmethod
    def split_train_test(data: np.ndarray, train_ratio=0.8) -> tuple[np.ndarray, np.ndarray]:
        np.random.shuffle(data)

        train_size: int = int(np.floor(data.shape[0] * train_ratio))
        return data[:train_size, :], data[train_size:, :]
