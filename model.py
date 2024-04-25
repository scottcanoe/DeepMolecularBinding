import torch


class Model(torch.nn.Module):
    """
    Feedforward model for estimating protein binding from molecular embeddings.
    Note that this model is kept separate from the embedding model for flexibility,
    which is located in `embedding.py`.
    """

    def __init__(self):
        super(Model, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(76800, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 250),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(250, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 3),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x
