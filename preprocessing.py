import h5py
import numpy as np
from common import *

def make_training_binds():
    df = load_data('train', columns=['binds'])
    binds = df['binds'].values
    binds = binds.reshape(-1, 3)
    path = DATADIR / 'training_binds.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=binds)

def check_training_structure():
    """
    training data: Organized into 3-row chunks, where each chunk corresponds to the same molecule and 3 different proteins.
    Proteins are in order BRD4, HSA, sEH. This has been checked and verified.

    Want to combine training data with one molecule and proteins binding indicators into a 3-vector.
    """

    df = load_data('train', columns=['molecule_smiles', 'protein_name', 'binds'])
    unique = df['molecule_smiles'].unique()

    n_unique = len(unique)
    protein_names = np.array(['BRD4', 'HSA', 'sEH'])

    for i in range(n_unique):
        d = df.iloc[i * 3: (i + 1) * 3]
        smiles = d['molecule_smiles'].unique()
        if len(smiles) != 1:
            print('not unique')
        proteins = d['protein_name']
        if not np.array_equal(proteins, protein_names):
            print('not equal names')
