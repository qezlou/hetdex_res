"""
Helper to plot the fibers and the covariance matrices.
"""
import logging
import h5py
import numpy as np
from os import path as op
import sys
import matplotlib.pyplot as plt


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
