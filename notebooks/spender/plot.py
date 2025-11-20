"""
Plot spender plots
"""
import h5py
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
        self.recon_path = op.join(data_dir, 'recon', recon_file)
        self.losses = np.array(torch.load(model_path, map_location="cpu")['losses'])

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
        with h5py.File(self.recon_path, 'r') as f:
            train_latents = f['train_latents'][:]

        fig = corner.corner(train_latents, labels=[f"z{i}" for i in range(train_latents.shape[1])],
                            show_titles=True, title_fmt=".2f",
                            title_kwargs={"fontsize": 12})
    
    def recon_og(self):

        with h5py.File(self.recon_path, 'r') as f:
            og_spectra = f['train_og_spectra'][:]
            recon_spectra = f['train_recon_spectra'][:]
            print(f'number of spectra: {og_spectra.shape[0]}')

        #ind_args_sort = np.argsort(np.mean(abs_diff))
        #ind_sel = ind_args_sort[-5:]  # worst 5 reconstructions
        # select indices from the top 5% of mean absolute differences
        threshold = np.percentile(np.abs(recon_spectra), 99)
        top_inds = np.where(np.abs(recon_spectra) >= threshold)[0]

        if top_inds.size == 0:
            # fallback: random selection if no indices found (very unlikely)
            ind_sel = np.random.randint(0, og_spectra.shape[0], size=5)
        else:
            # sample 5 indices from the top set (allow replacement only if fewer than 5)
            replace = top_inds.size < 5
            ind_sel = np.random.choice(top_inds, size=5, replace=replace)


        #ind_sel = np.random.randint(0, og_spectra.shape[0], size=5)
        
        
        for i, ind in enumerate(ind_sel):
            fig, axs = plt.subplots(1, 1, figsize=(20, 5))
            axs.plot(og_spectra[ind], label='Original', alpha=0.7)
            axs.plot(recon_spectra[ind], label='Reconstructed', alpha=0.7)
            axs.legend(frameon=False)
            axs.set_ylabel("Flux")
            axs.set_ylim((-0.6, 0.6))
            axs.grid(which='both', linestyle='--', alpha=0.8)
            axs.set_xlabel("Wavelength Bin")
    
    def latent_umap(self):
        """Plot UMAP projection of latent space.

        """
        with h5py.File(self.recon_path, 'r') as f:
            train_latents = f['train_latents'][:]

        reducer = umap.UMAP()
        embedding_train = reducer.fit_transform(train_latents)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(embedding_train[:, 0], embedding_train[:, 1], s=1, label='Train', alpha=0.5)
        if self.validloader is not None:
            embedding_valid = reducer.transform(self.latents_valid)
            ax.scatter(embedding_valid[:, 0], embedding_valid[:, 1], s=1, label='Valid', alpha=0.5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()