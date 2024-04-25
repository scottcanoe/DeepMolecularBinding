from os import PathLike
from pathlib import Path
import sys
from time import perf_counter as clock
from typing import Sequence, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

__all__ = [
    'DATADIR',
    'DEVICE',
    'load_data',
]

DATADIR = Path('/home/gavorniklab/Scott/data/leash')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(name: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = DATADIR / f'{name}.parquet'
    df = pd.read_parquet(path, columns=columns)
    return df
