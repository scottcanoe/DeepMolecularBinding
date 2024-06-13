from os import PathLike
from pathlib import Path
from time import perf_counter as clock
from typing import Sequence, Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import AutoModel, AutoTokenizer

from common import DATADIR, DEVICE, load_data




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

        # Load transformer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.model = self.model.to(DEVICE)


    def get_training_batch(
            self,
            batch_size: Optional[int] = None,
    ):

        return self.get_batch('training', batch_size=batch_size)

    def get_validation_batch(
            self,
            batch_size: Optional[int] = None,
    ):
        """
        Probably actually want the validation set to not have a balanced split.

        :param batch_size:
        :param shuffle:
        :return:
        """
        return self.get_batch('validation', batch_size=batch_size)

    def get_batch(
            self,
            kind: str,
            batch_size: Optional[int] = None,
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
        true_indices_batch = np.random.choice(true_indices, half_batch_size, replace=True)
        false_indices_batch = np.random.choice(false_indices, half_batch_size, replace=True)

        # Get smiles and binding vectors for training data.
        true_smiles = list(self.smiles[true_indices_batch])
        true_y = self.binds[true_indices_batch]
        false_smiles = list(self.smiles[false_indices_batch])
        false_y = self.binds[false_indices_batch]

        # Embed smiles.
        true_tokens = self.tokenizer(true_smiles, padding=True, return_tensors="pt")
        false_tokens = self.tokenizer(false_smiles, padding=True, return_tensors="pt")
        true_tokens = true_tokens.to(DEVICE)
        false_tokens = false_tokens.to(DEVICE)
        with torch.no_grad():
            true_outputs = self.model(**true_tokens)
            false_outputs = self.model(**false_tokens)
        true_X = true_outputs.pooler_output
        false_X = false_outputs.pooler_output

        # Combine input and output data.
        X = torch.cat([true_X, false_X], dim=0)
        Y = torch.tensor(np.concatenate([true_y, false_y]), dtype=torch.float32).to(DEVICE)
        return X, Y

class Model(torch.nn.Module):
    """
    Feedforward model for estimating protein binding from molecular embeddings.

    """

    def __init__(self):
        super(Model, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 300),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(300, 3),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

def save_state(
    step: Union[int, str],
    model: torch.nn.Module,
    train_losses: Sequence,
    val_losses: Sequence,
    train_steps: Sequence,
    val_steps: Sequence,
) -> None:

    save_dir = DATADIR / f'hf_states/{step}'
    save_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), save_dir / 'model.pth')
    np.save(save_dir / 'train_losses.npy', train_losses)
    np.save(save_dir / 'val_losses.npy', val_losses)
    np.save(save_dir / 'train_steps.npy', train_steps)
    np.save(save_dir / 'val_steps.npy', val_steps)

def training_loop():
    batch_size = 1000
    n_steps = 50_001
    validation_size = 0.05
    plot_every = 1000
    save_every = 1000
    validate_every = 50

    loader = RandomBatchLoader(batch_size=batch_size, validation_size=validation_size)
    val_X, val_Y = loader.get_validation_batch(batch_size=batch_size)

    model = Model().to(DEVICE)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    t_start = clock()
    train_losses, val_losses = [], []
    train_steps, val_steps = [], []
    for step in range(n_steps):

        X, Y = loader.get_training_batch()
        pred = model(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_steps.append(step)
        if step % validate_every == 0:
            with torch.no_grad():
                pred = model(val_X)
                loss = loss_fn(pred, val_Y)
                val_losses.append(loss.item())
                val_steps.append(step)

        if plot_every is not None and step % plot_every == 0 and step > 0:
            fig, ax = plt.subplots()
            ax.plot(train_steps, train_losses, color='black')
            ax.plot(val_steps, val_losses, color='red')
            plt.show()

        if step % save_every == 0:
            save_state(step, model, train_losses, val_losses, train_steps, val_steps)

        print(f'Epoch {step}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

    # Finally, save the state and plot final losses.
    fig, ax = plt.subplots()
    ax.plot(train_losses, color='black')
    ax.plot(val_losses, color='red')
    plt.show()
    save_state('last', model, train_losses, val_losses, train_steps, val_steps)


def make_submission(state_dir: PathLike) -> None:

    state_dir = Path(state_dir)

    """
    Make predictions on state data.
    """
    # Load transformer
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct",
        trust_remote_code=True,
    )
    embedder = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct",
        deterministic_eval=True,
        trust_remote_code=True,
    )

    embedder = embedder.to(DEVICE)
    # - load model and embedder.

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(state_dir / 'model.pth'))
    model.eval()

    # - predict bindings one molecule at a time.
    test_df = load_data('test', columns=['id', 'molecule_smiles', 'protein_name'])
    unique = list(test_df['molecule_smiles'].unique())
    molecule_smiles, protein_name, binds = [], [], []

    true_smiles = unique[:3]

    with torch.no_grad():
        for i, sm in enumerate(unique):
            tokens = tokenizer([sm], padding=True, return_tensors="pt").to(DEVICE)
            X = embedder(**tokens).pooler_output
            pred = model(X)[0].cpu().numpy()
            molecule_smiles.extend([sm, sm, sm])
            protein_name.extend(['BRD4', 'HSA', 'sEH'])
            binds.extend(pred)
            print(f'Predicted {i}/{len(unique)} -- {100*i/len(unique):.2f}%', end='\r')

    preds = pd.DataFrame({'molecule_smiles': molecule_smiles, 'protein_name': protein_name, 'binds': binds})
    preds.to_csv(state_dir / 'test_predictions.csv', index=False)

    """
    Create the maps needed to connect test rows to predicted data.
    """

    # - A dictionary that maps id -> (smiles, protein).
    ids = test_df['id'].values
    col1 = test_df['molecule_smiles'].values
    col2 = test_df['protein_name'].values
    info = list(zip(col1, col2))
    id_to_info = dict(zip(ids, info))

    # - A dictionary that maps (smiles, protein) -> binds.
    col1 = preds['molecule_smiles'].values
    col2 = preds['protein_name'].values
    info = list(zip(col1, col2))
    binds = preds['binds'].values
    info_to_binds = dict(zip(info, binds))

    # - Now create a map from id -> binds.
    ids, binds = [], []
    for id, info in id_to_info.items():
        b = info_to_binds[info]
        binds.append(b)
        ids.append(id)

    # - Save the submission.
    submission = pd.DataFrame({'id': ids, 'binds': binds})
    submission.to_csv(state_dir / 'submission.csv', index=False)
