from typing import Optional
import h5py
import numpy as np
import torch
from common import load_data, DATADIR, DEVICE
from embedding import Embedder


class RandomBatchLoader:
    """
    Class that delivers random, balanced, batches of (embedded) training and validation data.

    """

    def __init__(
            self,
            batch_size: int = 1000,
            validation_size: float = 0.05,
            random_init: bool = True,
    ):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.validation_size = validation_size

        # Load unique smiles.
        df = load_data('train', columns=['molecule_smiles'])
        self.smiles = df['molecule_smiles'].values.reshape(-1, 3)[:, 0]

        # Load binding output vector (precomputed) corresponding to unique smiles.
        with h5py.File(DATADIR / 'training_binds.h5', 'r') as f:
            self.binds = f['data'][:]

        # Find indices of binding and non-binding examples.
        sums = self.binds.sum(axis=1)
        self.true_indices = np.where(sums > 0)[0]
        self.false_indices = np.where(sums == 0)[0]
        if random_init:
            np.random.shuffle(self.true_indices)
            np.random.shuffle(self.false_indices)

        # Split indices into training and validation sets.
        split_point = round(len(self.true_indices) * (1 - self.validation_size))
        self.true_indices_train = self.true_indices[:split_point]
        self.true_indices_val = self.true_indices[split_point:]

        split_point = round(len(self.false_indices) * (1 - self.validation_size))
        self.false_indices_train = self.false_indices[:split_point]
        self.false_indices_val = self.false_indices[split_point:]

        # Load embedder.
        self.embedder = Embedder()

    def get_training_batch(
            self,
            batch_size: Optional[int] = None,
            shuffle: bool = True,
    ):

        return self.get_batch('training', batch_size=batch_size, shuffle=shuffle)

    def get_validation_batch(
            self,
            batch_size: Optional[int] = None,
            shuffle: bool = True,
    ):
        """
        Probably actually want the validation set to not have a balanced split.

        :param batch_size:
        :param shuffle:
        :return:
        """
        return self.get_batch('validation', batch_size=batch_size, shuffle=shuffle)

    def get_batch(
            self,
            kind: str,
            batch_size: Optional[int] = None,
            shuffle: bool = True,
    ):
        assert kind in ['training', 'validation']
        batch_size = batch_size or self.batch_size
        half_batch_size = int(batch_size / 2)

        if kind == 'training':
            true_indices = self.true_indices_train
            false_indices = self.false_indices_train
        else:
            true_indices = self.true_indices_val
            false_indices = self.false_indices_val

        # Get random indices for binding and non-binding data from training set.
        true_indices_batch = np.random.choice(true_indices, half_batch_size, replace=False)
        false_indices_batch = np.random.choice(false_indices, half_batch_size, replace=False)

        # Get smiles and binding vectors for training data.
        true_smiles = self.smiles[true_indices_batch]
        true_y = self.binds[true_indices_batch]
        false_smiles = self.smiles[false_indices_batch]
        false_y = self.binds[false_indices_batch]

        # Embed smiles.
        true_X = self.embedder.embed(true_smiles)
        false_X = self.embedder.embed(false_smiles)

        # Combine input and output data.
        X, Y = np.concatenate([true_X, false_X]), np.concatenate([true_y, false_y])
        if shuffle:
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

        # Convert to torch tensors.
        X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        X, Y = X.to(DEVICE), Y.to(DEVICE)

        return X, Y
