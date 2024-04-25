from time import perf_counter as clock
from typing import Sequence

from matplotlib import pyplot as plt
import numpy as np
import torch

from common import DATADIR, DEVICE
from data_loader import RandomBatchLoader
from model import Model


def save_state(
    epoch: int,
    model: torch.nn.Module,
    train_losses: Sequence,
    val_losses: Sequence,
) -> None:

    save_dir = DATADIR / f'states/{epoch}'
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'model.pth')
    np.save(save_dir / 'train_losses.npy', train_losses)
    np.save(save_dir / 'val_losses.npy', val_losses)

def train_model(
    n_epochs,
    batch_size: int = 1_000,
    plot_losses: bool = True,
    save_every: int = 1_000,
    plot_every: int = 250,
) -> None:


    validation_size = 0.05

    loader = RandomBatchLoader(batch_size=batch_size, validation_size=validation_size)
    val_X, val_Y = loader.get_validation_batch(batch_size=1_000)

    model = Model().to(DEVICE)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    t_start = clock()
    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        X, Y = loader.get_training_batch(batch_size=batch_size)
        pred = model(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        with torch.no_grad():
            pred = model(val_X)
            loss = loss_fn(pred, val_Y)
            val_losses.append(loss.item())

        if epoch % save_every == 0:
            save_state(epoch, model, train_losses, val_losses)

        if plot_every is not None and epoch % plot_every == 0 and epoch > 0:
            if plot_losses:
                fig, ax = plt.subplots()
                ax.plot(train_losses, color='black')
                ax.plot(val_losses, color='red')
                plt.show()

        print(f'Epoch {epoch}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

    # Finally, save the state and plot final losses.
    save_state(epoch, model, train_losses, val_losses)
    fig, ax = plt.subplots()
    ax.plot(train_losses, color='black')
    ax.plot(val_losses, color='red')
    plt.show()

