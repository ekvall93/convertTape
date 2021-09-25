from tape.datasets import PrositFragmentationDatasetHCD
from typing import List, Union, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

def getDataLoader(lmdb, data_type, num_workers=1, batch_size=256):
    dataset = PrositFragmentationDatasetHCD(lmdb, data_type)
    return DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,  # type: ignore
        batch_size=batch_size)
