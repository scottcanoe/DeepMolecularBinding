from os import PathLike
from typing import Sequence

import numpy as np
import torch

from MolecularTransformerEmbeddings.load_data import ALPHABET_SIZE, EXTRA_CHARS
from MolecularTransformerEmbeddings.transformer import Transformer, create_masks
from common import DATADIR, DEVICE


class Embedder:

    """
    Class that embeds SMILES strings using pre-trained transformer model.

    Wraps the `Transformer` class from `MolecularTransformerEmbeddings`.

    """
    MAX_LENGTH: int = 150           # max length of SMILES string.
    EMBEDDING_SIZE: int = 512       # size of the embedding dimension.
    NUM_LAYERS: int = 6             # number of layers in the transformer.

    def __init__(
        self,
        checkpoint_path: PathLike = DATADIR / 'embedder/embedder.ckpt',
    ):

        print("Initializing Embedder...")
        self.model = Transformer(ALPHABET_SIZE, self.EMBEDDING_SIZE, self.NUM_LAYERS).eval()
        self.model = torch.nn.DataParallel(self.model)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.module
        self.encoder = self.model.encoder
        print("Embedder Initialized.")

    @staticmethod
    def encode_char(c: str) -> int:
        return ord(c) - 32

    @staticmethod
    def encode_smiles(string: str, start_char: str = EXTRA_CHARS['seq_start']) -> torch.Tensor:
        ints = [ord(start_char)] + [Embedder.encode_char(c) for c in string]
        return torch.tensor(ints, dtype=torch.long)[:Embedder.MAX_LENGTH].unsqueeze(0)

    def embed(self, smiles: Sequence[str]) -> np.ndarray:
        out = np.zeros([len(smiles), self.MAX_LENGTH, self.EMBEDDING_SIZE])
        with torch.no_grad():
            for i, sm in enumerate(smiles):
                encoded = self.encode_smiles(sm)  # one-hot encoding
                encoded = encoded.to(DEVICE)
                mask = create_masks(encoded)      # mask unused tokens
                embedding = self.encoder(encoded, mask)[0]
                embedding = embedding.cpu().numpy()
                out[i, 0:embedding.shape[0], :] = embedding
        return out

    def __call__(self, smiles: Sequence[str]) -> np.ndarray:
        return self.embed(smiles)

