"""
Plot spender plots
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt
import corner
import torch
import umap

import os.path as op

class HetSpenderPlot():

    def __init__(self, data_dir, model_file, recon_file, which='both'):
        self.data_dir = data_dir
        self.model_file = model_file
        self.recon_path = op.join(data_dir, 'recon', recon_file)
        

    def get_latents(self):
        with h5py.File(self.recon_path, 'r') as f:
            train_latents = f['train_latents'][:]
        return train_latents

    def plot_loss(self):
        """Plot training and validation loss curves."""
        model_path = op.join(self.data_dir, 'models', self.model_file)
        self.losses = np.array(torch.load(model_path, map_location="cpu")['losses'])
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(self.losses[:, 0], label="Training Loss")
        ax.plot(self.losses[:, 1], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_yscale('log')

    def plot_latent_corner(self):
        """Plot corner plot of latent space.

        """
        with h5py.File(self.recon_path, 'r') as f:
            train_latents = f['train_latents'][:]

        fig = corner.corner(train_latents, labels=[f"z{i}" for i in range(train_latents.shape[1])],
                            show_titles=True, title_fmt=".2f",
                            title_kwargs={"fontsize": 12})
    
    def recon_og_large_recon(self, recon_files=None, n_top=5):
        if recon_files is None:
            recon_paths = [self.recon_path]
        else:
            recon_paths = [op.join(self.data_dir, 'recon', rf) for rf in recon_files]

        # Load original spectra ONCE (from the first file)
        with h5py.File(recon_paths[0], 'r') as f:
            og_spectra = f['train_og_spectra'][:]
        print(f"Number of OG spectra: {og_spectra.shape[0]}")

        # Load reconstructed spectra from ALL files
        recon_list = []
        for path in recon_paths:
            with h5py.File(path, 'r') as f:
                recon_list.append(f['train_recon_spectra'][:])
                print(f"{path}: recon spectra: {f['train_recon_spectra'].shape[0]}")

        # We assume all recon files correspond to same number of samples
        n = og_spectra.shape[0]

        # Compute threshold based on ANY of the recon sets (use first file)
        recon_spectra_0 = recon_list[0]
        threshold = np.percentile(np.abs(recon_spectra_0), 99)
        top_inds = np.where(np.abs(recon_spectra_0) >= threshold)[0]

        if top_inds.size == 0:
            ind_sel = np.random.randint(0, n, size=n_top)
        else:
            replace = top_inds.size < 5
            ind_sel = np.random.choice(top_inds, size=n_top, replace=replace)

        # Plot OG + all recon versions for each selected sample
        for ind in ind_sel:
            fig, ax = plt.subplots(figsize=(20, 5))

            # Plot original
            ax.plot(og_spectra[ind], label="Original", alpha=0.8, linewidth=2)

            # Plot reconstructed versions (one per file)
            for k, recon_spectra in enumerate(recon_list):
                ax.plot(
                    recon_spectra[ind],
                    label=f"Reconstructed #{k+1}",
                    alpha=0.6,
                    lw=4 
                )

            ax.legend(frameon=False)
            ax.set_ylabel("Flux")
            ax.set_ylim((-0.6, 0.6))
            ax.grid(which='both', linestyle='--', alpha=0.8)
            ax.set_xlabel("Wavelength Bin")
            ax.set_title(f"Spectrum index {ind}")


    def recon_og_ind(self, ind_to_use):

        with h5py.File(self.recon_path, 'r') as f:
            og_spectra = f['train_og_spectra'][ind_to_use,:]
            recon_spectra = f['train_recon_spectra'][ind_to_use,:]
            print(f'number of spectra: {og_spectra.shape[0]}')

        replace = recon_spectra.shape[0] < 5
        ind_sel = np.random.choice(ind_to_use.size, size=5, replace=replace)

        
        for i, ind in enumerate(ind_sel):
            fig, axs = plt.subplots(1, 1, figsize=(20, 5))
            axs.plot(og_spectra[ind], label='Original', alpha=0.7)
            axs.plot(recon_spectra[ind], label='Reconstructed', alpha=0.7)
            axs.legend(frameon=False)
            axs.set_ylabel("Flux")
            axs.set_ylim((-0.6, 0.6))
            axs.grid(which='both', linestyle='--', alpha=0.8)
            axs.set_xlabel("Wavelength Bin")


    
    def latent_umap(self, frac_sample=1.0):
        """Plot UMAP projection of latent space.

        """
        if frac_sample < 1.0:
            all_latents = self.get_latents()
            n_samples = all_latents.shape[0]
            n_select = int(frac_sample * n_samples)
            select_inds = np.random.choice(n_samples, size=n_select, replace=False)
            train_latents = all_latents[select_inds, :]
        else:
            train_latents = self.get_latents()

        reducer = umap.UMAP()
        embedding_train = reducer.fit_transform(train_latents)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(embedding_train[:, 0], embedding_train[:, 1], s=1, label='Train', alpha=0.5)
        if hasattr(self, 'latents_valid'):
            embedding_valid = reducer.transform(self.latents_valid)
            ax.scatter(embedding_valid[:, 0], embedding_valid[:, 1], s=1, label='Valid', alpha=0.5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend()