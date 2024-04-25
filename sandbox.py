from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from common import DATADIR, DEVICE, load_data
from embedding import Embedder
from model import Model


def make_submission(state_dir: PathLike) -> None:

    state_dir = Path(state_dir)

    """
    Make predictions on state data.
    """
    # - load model and embedder.
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(state_dir / 'model.pth'))
    model.eval()
    embedder = Embedder()

    # - predict bindings one molecule at a time.
    test_df = load_data('test', columns=['id', 'molecule_smiles', 'protein_name'])
    unique = test_df['molecule_smiles'].unique()
    molecule_smiles, protein_name, binds = [], [], []
    with torch.no_grad():
        for sm in unique:
            X = embedder.embed([sm])
            X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            pred = model(X)
            pred = pred[0].cpu().numpy()
            molecule_smiles.extend([sm, sm, sm])
            protein_name.extend(['BRD4', 'HSA', 'sEH'])
            binds.extend(pred)

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



# state_dir = DATADIR / 'states/14999'
# train_losses = np.load(state_dir / 'train_losses.npy')
# val_losses = np.load(state_dir / 'val_losses.npy')
# fig, ax = plt.subplots()
# ax.plot(train_losses, color='black', label='trainining')
# ax.plot(val_losses, color='red', label='validation')
# ax.legend()
# ax.set_xlabel('iteration')
# ax.set_ylabel('loss')
# ax.set_xlim(-10, 15000)
# plt.savefig('docs/losses.png')
# plt.show()
