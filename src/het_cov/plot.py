"""
Helper to plot the fibers and the covariance matrices.
"""
import logging
import h5py
import numpy as np
from os import path as op
import sys
import matplotlib.pyplot as plt
import json


class Plot():
    """
    Class to plot the fibers and the covariance matrices.
    """
    def __init__(self, data_dir='/home/qezlou/HD1/data_het/data/emmission/', logging_level='INFO'):
        """
        Parameters
        ----------
        data_dir: str
            Directory where the h5 file with shotid list is located
        """
        self.logger = self.configure_logging(logging_level=logging_level, logger_name='Fibers')
        self.data_dir = data_dir
        with h5py.File(f'{data_dir}/wave.h5', 'r') as f:
            self.wave = f['wave'][:]

    def configure_logging(self, logging_level='INFO', logger_name='Fibers'):
        """
        Set up logging based on the provided logging level in an MPI environment.

        Parameters
        ----------
        logging_level : str, optional
            The logging level (default is 'INFO').
        logger_name : str, optional
            The name of the logger (default is 'BaseGal').

        Returns
        -------
        logger : logging.Logger
            Configured logger instance.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def get_cov(self, cov_file):

        with h5py.File(op.join(self.data_dir, cov_file), 'r') as f:
            shotids = f['shotid'][:]
            cov = f['cov_calfib_ffsky'][:]
        return shotids, cov

    def get_corr(self, cov_file):

        shotids, cov_matrices = self.get_cov(cov_file)
        corr_matrices = np.zeros_like(cov_matrices)
        for i in range(corr_matrices.shape[0]):
            # Compute standard deviations of each pixel
            std_devs = np.sqrt(np.diag(cov_matrices[i]))
        
            # Avoid division by zero
            std_devs[std_devs == 0] = 1e-10
        
            # Compute correlation matrix
            corr_matrices[i] = cov_matrices[i] / np.outer(std_devs, std_devs)
            
        return shotids, corr_matrices
    

    def cov_corr_rand_dateshots(self, cov_file, n=5):
        """
        Plot random covariance and correlation matrices from the given file.

        Parameters
        ----------
        cov_file : str
            The name of the HDF5 file containing covariance matrices.
        n : int
            Number of random matrices to plot.
        """

        shotids, cov = self.get_cov(cov_file)
        _, corr = self.get_corr(cov_file)
    
        fig, ax = plt.subplots(n, 2, figsize=(8, 3*n))
        ind_rand = np.random.randint(0, shotids.size, n)

        N = self.wave.shape[0]  # assuming square covariance matrices of shape (N, N)

        # Pick ~5 log-spaced ticks across the full wave range
        num_ticks = 5
        tick_indices = np.linspace(0, N - 1, num_ticks, dtype=int)
        tick_positions = [self.wave[i] for i in tick_indices]
        tick_labels = [f"{self.wave[i]:.0f}" for i in tick_indices]

        # Loop over each shot
        for i, ind in enumerate(ind_rand):
            ax[i, 0].set_title(f'{shotids[ind]}-cov')
            ax[i, 1].set_title(f'{shotids[ind]}-corr')

            # Show images with wavelength-scaled axes using 'extent'
            im0 = ax[i, 0].imshow(cov[ind, :, :], origin='lower', cmap='viridis',
                                aspect='auto', extent=[self.wave[0], self.wave[-1], self.wave[0], self.wave[-1]])
            im1 = ax[i, 1].imshow(corr[ind, :, :], origin='lower', cmap='viridis', vmin=0, vmax=1,
                                aspect='auto', extent=[self.wave[0], self.wave[-1], self.wave[0], self.wave[-1]])

            # Set wavelength ticks and labels
            for a in [ax[i, 0], ax[i, 1]]:
                a.set_xticks(tick_positions)
                a.set_xticklabels(tick_labels)
                a.set_yticks(tick_positions)
                a.set_yticklabels(tick_labels)
                a.set_xlabel("Wavelength")
                a.set_ylabel("Wavelength")

            fig.colorbar(im0, ax=ax[i, 0], fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=ax[i, 1], fraction=0.046, pad=0.04)
        fig.tight_layout()

    
    def corr_rand_dateshots(self, cov_file, n=5):
        """
        Plot random correlation matrices from the given file.

        Parameters
        ----------
        cov_file : str
            The name of the HDF5 file containing covariance matrices.
        n : int
            Number of random matrices to plot.
        """

        shotids, corr = self.get_corr(cov_file)
    
        fig, axes = plt.subplots(n, n, figsize=(n*4, n*4))
        ind_rand = np.random.randint(0, shotids.size, 25)

        N = self.wave.shape[0]  # assuming square covariance matrices of shape (N, N)
        # Pick ~5 log-spaced ticks across the full wave range
        num_ticks = 5
        tick_indices = np.linspace(0, N - 1, num_ticks, dtype=int)
        tick_positions = [self.wave[i] for i in tick_indices]
        tick_labels = [f"{self.wave[i]:.0f}" for i in tick_indices]

        # Flatten the axes array for easier iteration
        axes_flat = axes.flatten()

        # Loop over each shot
        for i, ind in enumerate(ind_rand):
            # Plot correlation matrix
            im = axes_flat[i].imshow(corr[ind, :, :], origin='lower', cmap='viridis', vmin=0, vmax=1,
                                aspect='auto', extent=[self.wave[0], self.wave[-1], self.wave[0], self.wave[-1]])

            axes_flat[i].set_title(f'Shot ID: {shotids[ind]}')
            
            # Set wavelength ticks and labels
            axes_flat[i].set_xticks(tick_positions)
            axes_flat[i].set_xticklabels(tick_labels, rotation=45)
            axes_flat[i].set_yticks(tick_positions)
            axes_flat[i].set_yticklabels(tick_labels)
            
            # Only add axis labels on the edge plots
            if i % 5 == 0:  # Left edge
                axes_flat[i].set_ylabel("Wavelength (Å)")
            if i >= 20:  # Bottom edge
                axes_flat[i].set_xlabel("Wavelength (Å)")

        # Add colorbar to each row, positioned at the right side
        for row in range(5):
            # Get the last plot in the row
            last_ax_in_row = axes[row, -1]
            # Create a new axis for the colorbar at the right of the last plot
            cbar_ax = fig.add_axes([last_ax_in_row.get_position().x1 + 0.01, 
                                last_ax_in_row.get_position().y0,
                                0.015,
                                last_ax_in_row.get_position().height])
            # Add the colorbar
            fig.colorbar(im, cax=cbar_ax, label='Correlation')

        plt.tight_layout()
        # Adjust layout to make room for colorbars
        plt.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)

    
    def median_corr(self, cov_file):
        """
        Plot the median correlation matrix from the given file.

        Parameters
        ----------
        cov_file : str
            The name of the HDF5 file containing covariance matrices.
        """
        _, corr = self.get_corr(cov_file)
        # Create a figure for the median and standard deviation of correlation matrices
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))

        # Calculate the median correlation matrix across all shots
        # Using nanmedian to handle any NaN values
        median_corr = np.nanmedian(corr, axis=0)

        # Calculate the standard deviation of correlation matrices
        std_corr = np.nanstd(corr, axis=0)

        # Plot the median correlation matrix
        im0 = ax[0].imshow(median_corr, origin='lower', cmap='viridis', vmin=0, vmax=1,
                        aspect='auto', extent=[self.wave[0], self.wave[-1], self.wave[0],self.wave[-1]])

        # Plot the standard deviation matrix
        im1 = ax[1].imshow(std_corr, origin='lower', cmap='viridis',
                        aspect='auto', extent=[self.wave[0], self.wave[-1], self.wave[0], self.wave[-1]])

        # Set titles and axis labels
        ax[0].set_title('Median Correlation Matrix Across All Shots')
        ax[0].set_xlabel('Wavelength (Å)')
        ax[0].set_ylabel('Wavelength (Å)')

        ax[1].set_title('Standard Deviation of Correlation Matrices')
        ax[1].set_xlabel('Wavelength (Å)')
        ax[1].set_ylabel('Wavelength (Å)')


        # Set wavelength ticks and labels
        N = self.wave.shape[0]  # assuming square covariance matrices of shape (N, N)

        # Pick ~5 log-spaced ticks across the full wave range
        num_ticks = 5
        tick_indices = np.linspace(0, N - 1, num_ticks, dtype=int)
        tick_positions = [self.wave[i] for i in tick_indices]
        tick_labels = [f"{self.wave[i]:.0f}" for i in tick_indices]

        for a in ax:
            a.set_xticks(tick_positions)
            a.set_xticklabels(tick_labels)
            a.set_yticks(tick_positions)
            a.set_yticklabels(tick_labels)

        # Add colorbars
        cbar0 = fig.colorbar(im0, ax=ax[0], label='Correlation')
        cbar1 = fig.colorbar(im1, ax=ax[1], label='Standard Deviation')

        plt.tight_layout()
        plt.show()

class PCA():
    """
    Class to plot the PCA results.
    """
    def __init__(self, data_dir='/home/qezlou/HD1/data_het/data/emmission/', data_file='pca.h5', logging_level='INFO'):
        """
        Parameters
        ----------
        data_dir: str
            Directory where the h5 file with shotid list is located
        """
        self.logger = self.configure_logging(logging_level=logging_level, logger_name='plot.PCA')
        self.data_dir = data_dir
        with h5py.File(f'{data_dir}/wave.h5', 'r') as f:
            self.wave = f['wave'][:]

        with h5py.File(op.join(data_dir, data_file), 'r') as f:
            self.components = f['components'][:]
            self.explained_variance = f['explained_variance'][:]
            self.explained_variance_ratio = f['explained_variance_ratio'][:]
            self.mean = f['mean_spectrum'][:]
            self.shotids = f['shotid'][:]
        
        self.logger.info(f'Loaded PCA results for {self.shotids.size} shots.')

    def configure_logging(self, logging_level='INFO', logger_name='Plot-PCA'):
        """
        Set up logging based on the provided logging level in an MPI environment.

        Parameters
        ----------
        logging_level : str, optional
            The logging level (default is 'INFO').
        logger_name : str, optional
            The name of the logger (default is 'BaseGal').

        Returns
        -------
        logger : logging.Logger
            Configured logger instance.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging_level)

        # Create a console handler with flushing
        console_handler = logging.StreamHandler(sys.stdout)

        # Include Rank, Logger Name, Timestamp, and Message in format
        formatter = logging.Formatter(
            f'%(name)s | %(asctime)s | %(levelname)s  |  %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        console_handler.setFormatter(formatter)

        # Ensure logs flush immediately
        console_handler.flush = sys.stdout.flush  

        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger

    def variance_vs_components(self):
        """
        Plot the explained variance ratio.
        """
        fig, ax = plt.subplots(figsize=(3,4))
        for i in range(self.explained_variance_ratio.shape[0]):
            ax.plot(np.arange(1, self.explained_variance_ratio.shape[1]+1), 
                    np.cumsum(self.explained_variance_ratio[i,:]))
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Var ratio')
        ax.set_title('var ratio')
        ax.grid()
        fig.tight_layout()

    def individual_eigenspectrum(self):
        """
        Plot the first 50 eigen-spectra.
        """
        # Plot the first 50 components
        fig, ax = plt.subplots(25,2, figsize=(12,50))
        rand_shots = np.random.randint(0, self.shotids.size, size=3)
        for i in rand_shots:
            for c in range(50):
                ax[c//2, c%2].plot(self.wave, np.sqrt(self.explained_variance[i,c])*self.components[i,c,:], alpha=0.5, label=self.shotids[i])
                ax[c//2, c%2].set_title(f'Component {c+1}')
                ax[c//2, c%2].set_ylim(-0.05, 0.1)
                ax[c//2, c%2].set_xlabel('Wavelength (Å)')
                ax[c//2, c%2].set_ylabel(r'$v \times V [erg/s/cm^2]$')
                ax[c//2, c%2].legend(frameon=False)
        fig.tight_layout()

    
    def project_onto_pca(self, spectrum, ind_shot, n_components=10):
        """
        Project a given spectrum onto the PCA components.

        Parameters
        ----------
        spectrum : array-like
            The input spectrum to project.
        ind_shot: int
            The index to self.shotids to query PCA eigenspectra from
        n_components : int
            Number of PCA components to use for projection.

        Returns
        -------
        projected_spectrum : array-like
            The reconstructed spectrum from the PCA components.
        """
        if n_components > self.components.shape[1]:
            raise ValueError(f"n_components {n_components} exceeds available components {self.components.shape[1]}")

        # Center the input spectrum by subtracting the mean spectrum
        centered_spectrum = spectrum - self.mean[ind_shot]

        # Project onto the first n_components PCA components
        projection = centered_spectrum @ self.components[ind_shot, :n_components].T

        # Reconstruct the spectrum from the projection
        reconstructed_spectrum = projection @ self.components[ind_shot, :n_components] + self.mean[ind_shot]

        return reconstructed_spectrum
    
    def orig_vs_recon(self, shotid, n_components=10, n_fibers=5):
        """
        NOTE: You need acccess to HETDEX-API to run this function.
        Plot the original vs reconstructed spectrum for a given shotid.

        Parameters
        ----------
        shotid : int
            The shotid to plot.
        n_components : int
            Number of PCA components to use for reconstruction.
        n_fibers : int
            Number of fibers to randomly select from the shot.
        """
        from . import fibers
        config = {
            "masking": {
                "bad_fibers": True,
                "bad_pixels": True,
                "strong_continuum": True
            },
            "cov_options": {
                "per": "shot",
                "method": "pca",
                "l": self.explained_variance.shape[0]
            }
            }
        fibs = fibers.Fibers(self.data_dir, 
                             config=config,
                             logging_level='INFO')

        if shotid not in self.shotids:
            raise ValueError(f"shotid {shotid} not found in the dataset.")

        ind_shot = np.where(self.shotids == shotid)[0][0]

        # Load the original fiber spectrum and subsample from it
        orig_fiber_specs = fibs.get_fibers_one_shot(shotid)['calfib_ffsky']
        ind_fib = np.random.randint(0, orig_fiber_specs.shape[0], n_fibers)
        orig_fiber_specs = orig_fiber_specs[ind_fib]

        # Project and reconstruct the spectrum
        recon_fiber_specs = []
        for i in range(n_fibers):
            recon_fiber_specs.append(self.project_onto_pca(orig_fiber_specs[i], ind_shot=ind_shot, n_components=n_components).squeeze())

        # Plot original vs reconstructed
        fig, ax = plt.subplots(n_fibers, 1, figsize=(10, n_fibers*3))
        for i in range(n_fibers):
            ax[i].plot(self.wave, orig_fiber_specs[i], label='Og', alpha=0.7)
            ax[i].plot(self.wave, recon_fiber_specs[i], label=f'Recon ({n_components} PCs)', alpha=0.7)

            ax[i].set_title(shotid)
            ax[i].set_xlabel('Wavelength (Å)')
            ax[i].set_ylabel('Flux')
            ax[i].legend()
            ax[i].grid()
        fig.tight_layout()
        return fig

