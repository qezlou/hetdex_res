"""
Plot spender plots
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt
import corner
import torch
import umap
from sklearn.preprocessing import LabelEncoder

import os.path as op

class HetSpenderPlot():

    def __init__(self, data_dir, spec_file, model_file, recon_file, which='both'):

        self.wave = np.arange(3600, 5301, 2)
        self.data_dir = data_dir
        self.spec_file = spec_file
        self.model_file = model_file
        self.recon_file = recon_file
        self.og_spec, self.spec_err, self.shotids, self.amps, self.fiber_ids, self.multiframes = self.get_og_spectra()
        self.latents, self.recon_spec, self.spec_inds = self.get_recon()
        ind_sort = np.argsort(self.spec_inds)
        self.latents =  self.latents[ind_sort, :]
        self.recon_spec = self.recon_spec[ind_sort, :]
        print(f"Original spectra shape: {self.og_spec.shape}")
    
    def get_og_spectra(self):
        with h5py.File(op.join(self.data_dir, 'fib_spec', self.spec_file), 'r') as f:
            print(f.keys())
            spec = f['calfib'][:, 65:916][:]
            err = f['calfibe'][:, 65:916][:]
            err[err <= 0] = np.inf  # masked pixels
            shotids = f['shotids'][:]
            amps = f['amps'][:]
            fiber_ids = f['fiber_ids'][:]
            multiframes = f['multiframes'][:]
        print(f'unique shotids in og spectra: {np.unique(shotids).size}')
        return spec, err, shotids, amps, fiber_ids, multiframes


    def get_recon(self):
        with h5py.File(op.join(self.data_dir, 'recon', self.recon_file), 'r') as f:
            latents = f['latents'][:]
            recon = f['recon_spectra'][:]
            spec_inds = f['inds'][:]
            print(f'saved recon shape: {recon.shape}')
        return latents, recon, spec_inds

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
        le = LabelEncoder()
        shotid_labels = le.fit_transform(self.shotids)  # self.shotids is array of shotids for self.latents
        fig = corner.corner(self.latents, labels=[f"z{i}" for i in range(self.latents.shape[1])],
                            show_titles=True, title_fmt=".2f",
                            title_kwargs={"fontsize": 12}, c=shotid_labels)
    
        #cbar = plt.colorbar(sc, ax=ax)
        #cbar.set_label("Shot ID (encoded)")
    
    def recon_og_large_recon(self, n_top=5, ind_sel=None):
        """Plot original and multiple reconstructed spectra for top varying samples.
        """
        if ind_sel is None:
            ind_sel = np.argsort(np.median(self.recon_spec, axis=1))[-n_top:]


        # Plot OG + all recon versions for each selected sample
        for ind in ind_sel:
            fig, ax = plt.subplots(figsize=(20, 5))

            # Plot original
            ax.plot(self.wave, self.og_spec[ind,:], label="Original", alpha=0.8, linewidth=2)
            ax.fill_between(self.wave, self.og_spec[ind,:] - self.spec_err[ind,:], self.og_spec[ind,:] + self.spec_err[ind,:], color='gray', alpha=0.3)
            ax.plot(self.wave, self.recon_spec[ind,:], label="Reconstructed", alpha=0.7)

            ax.legend(frameon=False)
            ax.set_ylabel("Flux")
            ax.set_ylim((-0.6, 0.6))
            ax.grid(which='both', linestyle='--', alpha=0.8)
            ax.set_xlabel("Wavelength Bin")
            ax.set_title(f"fiber_id: {self.fiber_ids[ind]} | array index :{ind}")


    
    def latent_umap(self, frac_sample=1.0):
        """Plot UMAP projection of latent space.

        """

        reducer = umap.UMAP()
        embedding_train = reducer.fit_transform(self.latents)
        # encode shotids -> integers -> colormap
        # Label encode
        le = LabelEncoder()
        amp_labels = le.fit_transform(self.amps)  # self.shotids is array of shotids for self.latents
        le = LabelEncoder()
        shotid_labels = le.fit_transform(self.shotids)  # self.shotids is array of shotids for self.latents

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sc_amps = ax[0].scatter(
            embedding_train[:, 0],
            embedding_train[:, 1],
            s=2,
            c=shotid_labels,
            cmap='Spectral',
            alpha=0.6
        )
        cbar = plt.colorbar(sc_amps, ax=ax[0])
        cbar.set_label("Shot ID (encoded)")


        sc_amps = ax[1].scatter(
            embedding_train[:, 0],
            embedding_train[:, 1],
            s=2,
            c=amp_labels,
            cmap='Spectral',
            alpha=0.6
        )
        cbar = plt.colorbar(sc_amps, ax=ax[1])
        cbar.set_label("Amplitude (encoded)")    

        for i in range(2):
            ax[i].set_xlabel("UMAP 1")
            ax[i].set_ylabel("UMAP 2")
            ax[i].set_title("UMAP colored by shotid")
        
        fig.tight_layout()