from typing import Tuple, Any
import torch
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame

class MyDataset(Dataset):
    def __init__(self, df: DataFrame) -> None:
        self.data = df
        self.features = [f'feature{i}' for i in range(107)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        features_tensor = torch.tensor(self.data.loc[idx, self.features].values, dtype=torch.float32)
        label = self.data.loc[idx, 'label']
        return features_tensor, label

class MyDataset_NoLabel(Dataset):
    def __init__(self, df: DataFrame) -> None:
        self.data = df
        self.features = [f'feature{i}' for i in range(106)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        features_tensor = torch.tensor(self.data.loc[idx, self.features].values, dtype=torch.float32)
        return features_tensor