"""
Plot spender plots
"""
import numpy as np
from spender import SpectrumAutoencoder, SpeculatorActivation
import torch
from matplotlib import pyplot as plt
import corner
from spender.data.hetdex import HETDEX
from spender import SpectrumAutoencoder, SpeculatorActivation
import umap
import os.path as op

class HetSpenderPlot():

    def __init__(self, data_dir, model_file, recon_file, which='both'):
        self.data_dir = data_dir
        model_path = op.join(data_dir, 'models', model_file)
        self.recon_file = recon_file
        self.loss = torch.load(model_path, map_location="cpu")['losses']

    def plot_loss(self):
        """Plot training and validation loss curves."""

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(self.losses[:, 0], label="Training Loss")
        ax.plot(self.losses[:, 1], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    def plot_latent_corner(self):
        """Plot corner plot of latent space.

        """

        fig = corner.corner(self.latents_train, labels=[f"z{i}" for i in range(self.latents_train.shape[1])],
                            show_titles=True, title_fmt=".2f",
                            title_kwargs={"fontsize": 12})
        if self.validloader is not None:
            corner.corner(self.latents_valid, fig=fig, color='C1', labels=[f"z{i}" for i in range(self.latents_valid.shape[1])],
                          show_titles=False)
    
    def plot_latent_umap(self):
        """Plot UMAP projection of latent space.

        """

        reducer = umap.UMAP()
        embedding_train = reducer.fit_transform(self.latents_train)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(embedding_train[:, 0], embedding_train[:, 1], s=1, label='Train', alpha=0.5)
        if self.validloader is not None:
            embedding_valid = reducer.transform(self.latents_valid)
            ax.scatter(embedding_valid[:, 0], embedding_valid[:, 1], s=1, label='Valid', alpha=0.5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()