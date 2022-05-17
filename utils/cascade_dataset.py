import os
import torch
from torch.utils.data import Dataset
from .dataset import TokenizedDataset

class CascadeDataset(TokenizedDataset):
    def __getitem__(self, index):
        item=super().__getitem__(index)
        if hasattr(self,'last_predictions'):
            item['last_predictions']=self.last_predictions[index]
        return item

