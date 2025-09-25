"""
Mostly based on what Mahan is doing for absorption
"""

from astropy.table import Table, join
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import FiberIndex
from hetdex_api.detections import Detections
from elixer import spectrum_utilities as SU
import numpy as np
import h5py
import os.path as op
from glob import glob
import time
import logging
import sys
from sklearn.decomposition import PCA
import json


class Fibers():
    """
    Getting all "good fibers" spectra for all shots in HDR5
    Should run on both JupyterHub and Compute nodes on Lonestar6
    """
    def __init__(self, data_dir= '/work2/06536/qezlou/hetdex/data/', 
                 config=None,
                 logging_level='INFO'):
        """
        Parameters
        ----------
        data_dir: str
            Directory where the h5 file with shotid list is located
        """
        self.logger = self.configure_logging(logging_level=logging_level, logger_name='Fibers')
        self.data_dir = data_dir
        self.save_dir = data_dir
        self.shotids_list = self._get_shotids_list()
        # Load masking options from config or use defaults
        self.masking = config.get(
            'masking', {
            'bad_fibers': True,
            'bad_pixels': True,
            'strong_continuum': True,
            'top_varying_pixels': True,
            "top_percent":5.,
            "top_fiber_frac":0.3
                            
            })
        # The calibrated flux to use: `self.calfib_type` or `calfib`
        self.calfib_type = config.get('calfib_type', 'calfib')
        # Check if we want to normalize the fibers
        self.normalize = config.get('normalize', False)
        # Load covariance options from config or use defaults
        self.cov_options = config.get('cov_options', {
            'per': 'shot',
            'method': 'pca',
            'l': 20
        })
        self.logger.info(f'Masking options: {self.masking}')
        self.logger.info(f'Covariance options: {self.cov_options}')
        self.logger.info(f'Flux, calfib_type: {self.calfib_type}')

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
    
    def _get_shotids_list(self):
        """
        Loading the list of shotids from HDR5 shotlist, for now it is 6771 in total
        """
        prefix= 'shotlist_*.txt'
        shotlist_files = sorted(glob(op.join(self.data_dir, prefix)))
        for i, f in enumerate(shotlist_files):
            self.logger.debug(f'loading shotlist from {f}')
            shotids = np.loadtxt(f, dtype=int)
            if i==0:
                shotids_list = shotids
            else:
                shotids_list = np.append(shotids_list, shotids)
        shotids_list = np.sort(shotids_list)[::-1]
        self.logger.info(f'we have {len(shotids_list)} shotids in total')
        return shotids_list

    def mask_top_varying_pixels(self, fib_tab):
        """
        Get the top 5th and 95th percentiles of flux values among all pixels and fibers.
        Mask pixels that have more than 0.3 of the fibers with a flux outside this range.
        The mask is applied by replacing the flux value with the median flux that individual fiber.
        Parameters
        ----------
        fib_tab: astropy Table
            Table with fiber_id, self.calfib_type and flag (True for good fibers)
        Returns
        -------
        fib_tab: astropy Table
            Table with fiber_id, self.calfib_type and flag (True for good fibers)
        """
        percentiles = [self.masking['top_percent'], 100-self.masking['top_percent']]
        flux_percentiles = np.percentile(fib_tab[self.calfib_type], percentiles)
        outlier_mask = (fib_tab[self.calfib_type] < flux_percentiles[0]) | (fib_tab[self.calfib_type] > flux_percentiles[1])
        outlier_fiber_counts = np.sum(outlier_mask, axis=0)
        outlier_fiber_ratio = outlier_fiber_counts / fib_tab[self.calfib_type].shape[0]
        top_varying_pixels = np.where(outlier_fiber_ratio > self.masking['top_fiber_frac'])[0]
        self.logger.info(f'Identified {len(top_varying_pixels)} top varying pixels')
        fib_tab[self.calfib_type][:, top_varying_pixels] = np.median(fib_tab[self.calfib_type], axis=1)[:, None]
        return fib_tab

    def get_fibers_one_shot(self, shotid):
        """
        Get fiber table for a single shot
        Parameters
        ----------
        shotid: int
            Shot ID
        Returns
        -------
        fib_tab: astropy Table
            Table with fiber_id, self.calfib_type and flag (True for good fibers)
        """
        # 1. Load the fluxes, "self.calfib_type", "fiber_id" to cross match for flags and "calfibe" to find bad fibers
        keys_to_query = ['fiber_id', self.calfib_type, 'calfibe']
        fibtable_one_shot = get_fibers_table(shot=shotid, survey='hdr5',
                                             verbose=False, add_rescor=False)[keys_to_query]
        F = FiberIndex(survey='hdr5') 
        fib_tab_findex = F.return_shot( shotid)['fiber_id','flag']
        fib_tab= join(fibtable_one_shot, fib_tab_findex, "fiber_id" )
        # 2. Only keep good fibers, flag=True
        self.logger.info(f'Total fibers: {len(fib_tab)}')
        fib_tab = fib_tab[fib_tab['flag']==True]

        if self.masking['bad_pixels']:
            # 3. Find the bad pixels for each fiber. this includes cosmic rays.
            # They are fibers with non-positive calfib error.
            # We replace the flux value by the median, but
            # NOTE: we may need to ignore these pixels altogether
            # when working on a probabilistic model
            mask_bad_pixs = fib_tab['calfibe'] <= 0
            fib_tab[self.calfib_type][mask_bad_pixs] = np.median(fib_tab[self.calfib_type][~mask_bad_pixs], axis=0)
            fib_tab.remove_column('calfibe')
            self.logger.info(f"Good fibers: {len(fib_tab)}, Fraction of good pixels {1 - np.sum(mask_bad_pixs)/fib_tab[self.calfib_type].size}")
            del mask_bad_pixs
        
        if self.masking['strong_continuum']:
            #4.  Remove strong continuum sources, from Mahan's code and Elixer
            wl = np.linspace(3470,5540,1036)
            wl_vac=SU.air_to_vac(wl)
            mask_zone1 = (3500 < wl_vac) & (wl_vac < 3860)
            mask_zone2 = (3860 < wl_vac) & (wl_vac < 4270)
            mask_zone3 = (4270 < wl_vac) & (wl_vac < 4860)
            mask_zone4 = (4860 < wl_vac) & (wl_vac < 5090)
            mask_zone5 = (5090 < wl_vac) & (wl_vac < 5500)

            medians = np.array([np.nanmedian(fib_tab[self.calfib_type][:, mask], axis=1) for mask in [mask_zone1, mask_zone2, mask_zone3, mask_zone4, mask_zone5]])
            ## Remove the fiber even if one of the regions has a high continuum
            # Define different upper bounds for the blue side
            upper_bounds = [0.25, 0.08, 0.08, 0.08, 0.08]  # example values, adjust as needed
            lower_bound = -0.02  # same lower bound for all

            valid_mask = np.ones(medians.shape[1], dtype=bool)
            for i, ub in enumerate(upper_bounds):
                valid_mask &= (medians[i] > lower_bound) & (medians[i] < ub)
            self.logger.info(f' Remaining fraction after removing continuum sources {np.sum(valid_mask)/len(fib_tab)} ')
            fib_tab = fib_tab[valid_mask]
            del medians, valid_mask

        if self.masking['top_varying_pixels']:
            # 4. Mask top varying pixels across fibers
            fib_tab = self.mask_top_varying_pixels(fib_tab)

        return fib_tab

    def get_normalized_fibers_one_shot(self, shotid):
        """
        Get normalized fiber table for a single shot.
        I = (I-<I>)(\lambda)/sigma_I(\lambda)
        Parameters
        ----------
        shotid: int
            Shot ID
        Returns
        -------
        fib_tab: astropy Table
            Table with fiber_id, self.calfib_type and flag (True for good fibers)
        """
        fib_tab = self.get_fibers_one_shot(shotid)
        # Normalize each fiber by its mean flux value
        means = np.mean(fib_tab[self.calfib_type], axis=0)
        stds = np.std(fib_tab[self.calfib_type], axis=0)
        fib_tab[self.calfib_type] = (fib_tab[self.calfib_type] - means[None, :])/stds[None, :]
        return fib_tab
    
    def get_fibers(self):
        """
        Iterate over all shotids and save the `self.calfib_type` spectra
        for each shotid in a separate h5 file.
        """
        for shotid in self.shotids_list:
            fib_tab = self.get_fibers_one_shot(shotid)
            with h5py.File(op.join(f'{self.calfib_type}_{shotid}.h5'), 'w') as fw:
                fw[self.calfib_type] = fib_tab[self.calfib_type]

    def get_cov(self, save_file=None):
        """
        Iterate over all shotids and compute the covariance matrix
        for the `self.calfib_type` spectra and save it in a separate h5 file.
        """
        if save_file is None:
            save_file = f'cov_{self.calfib_type}_rmvd_bad_fibs_cont.h5'
        cov_path = op.join(self.save_dir, save_file)
        
        if self.cov_options['per'] == 'shot':
            if self.cov_options['method'] == 'full':
                if op.exists(cov_path):
                    cov_all, shotids_in_cov = self.load_cov(cov_path)
                    if len(shotids_in_cov) != len(self.shotids_list):
                        shotids_remaining = np.setdiff1d(self.shotids_list, shotids_in_cov)
                        progress = len(shotids_in_cov)
                    else:
                        return cov_all, shotids_in_cov
                else:
                    shotids_remaining = self.shotids_list
                    progress = 0

                self.logger.info(f'{len(shotids_remaining)} shotids remaining to compute covariance for')
                for i, shotid in enumerate(shotids_remaining):
                    self.logger.info(f'working on shotid: {shotid}, progress {progress}/{len(self.shotids_list)}')
                    fib_spec = self.get_fibers_one_shot(shotid)[self.calfib_type]
                    if 'cov_all' in locals():
                        cov_all= np.append(cov_all, self.get_cov_one_shot(fib_spec)[None,:,:], axis=0)
                        shotids_in_cov = np.append(shotids_in_cov, shotid)
                    else:
                        cov_all = self.get_cov_one_shot(fib_spec)[None,:,:]
                        shotids_in_cov = np.array([shotid])[None,:]
                    if i%10 ==0:
                        self.save_cov(cov_path, cov_all, shotids_in_cov)
                    progress += 1
                self.save_cov(cov_path, cov_all, shotids_in_cov)
        else:
            self.logger.error('Currently only per shot covariance is implemented')
            raise NotImplementedError

        return cov_all, shotids_in_cov

    def get_cov_one_shot(self, fib_spec):
        """
        Compute the covariance matrix for a given set of fiber spectra.

        Parameters
        ----------
        fib_spec : np.ndarray, shape (N_fibers, N_wavelengths)
            Array containing the fiber spectra.
        Returns
        -------
        cov_matrix : np.ndarray, shape (N_wavelengths, N_wavelengths)
            Covariance matrix computed from the fiber spectra.
        """
        if self.cov_options['method'] == 'full':
            cov_matrix = np.cov(fib_spec, rowvar=False)
        elif self.cov_options['method'] == 'pca':
            if 'l' not in self.cov_options or self.cov_options['l'] is None:
                raise ValueError("The number of PCA components 'l' must be specified in cov_options.")
            n_components = self.cov_options['l']
            # Step 1: Fit PCA
            pca = PCA(n_components=n_components)
            X_proj = pca.fit_transform(fib_spec)            # shape: (n_fibers, k)
            X_approx = pca.inverse_transform(X_proj) # shape: (n_fibers, m)
            self.logger.info(f'Explained variance ratio by top {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}')

            # Step 3: Estimate covariance of the approximated data
            cov_matrix = np.cov(X_approx, rowvar=False)
        else:
            raise ValueError("cov_method must be either 'full' or 'PCA'.")
        
        return cov_matrix

    def save_cov(self, cov_path, cov_all, shotids_in_cov):
        """
        save the covariance matrix for a given shotid
        Parameters
        ----------
        cpv_path: str
            Path to save the covariance matrix
        cov_all: np.ndarray, shape (N_shots, N_wavelengths, N_wavelengths)
            Covariance matrix
        shotids_in_cov: np.ndarray, shape (N_shots,)
            Shot IDs corresponding to the covariance matrix
        """
        self.logger.info(f'saving cov in {cov_path}')
        with h5py.File(cov_path, 'w') as fw:
            fw[f'cov_{self.calfib_type}'] = cov_all
            fw['shotid'] = shotids_in_cov
        

    def load_cov(self, cov_path):
        """
        Load the covariance matrix for a given shotid
        Parameters
        ----------
        shotid: int
            Shot ID
        Returns
        -------
        cov: np.ndarray
            Covariance matrix
        """
        if op.exists(cov_path):
            with h5py.File(cov_path, 'r') as f:
                cov = f[f'cov_{self.calfib_type}'][:]
                shotids_in_cov = f['shotid'][:]
            return cov, shotids_in_cov
        else:
            self.logger.error(f'Covariance file {cov_path} does not exist.')
            return None

    def do_pca(self, save_file='pca.h5'):
        
        pca_path = op.join(self.save_dir, save_file)
        progress = 0
        if self.cov_options['per'] == 'shot':
            if op.exists(pca_path):
                comp_all, var_all, var_ratio_all, mean_all, shotids_in_pca = self.load_pca(pca_path)
                if comp_all.shape[0] != len(self.shotids_list):
                    shotids_remaining = np.setdiff1d(self.shotids_list, shotids_in_pca)
                    progress = len(shotids_in_pca)
                else:
                    return comp_all, var_all, var_ratio_all, mean_all, shotids_in_pca
            else:
                shotids_remaining = self.shotids_list
            self.logger.info(f'{len(shotids_remaining)} shotids remaining to compute covariance for')
            for i, shotid in enumerate(shotids_remaining):
                self.logger.info(f'working on shotid: {shotid}, progress {progress}/{len(self.shotids_list)}')
                if self.normalize:
                    fib_spec = self.get_normalized_fibers_one_shot(shotid)[self.calfib_type]
                else:
                    fib_spec = self.get_fibers_one_shot(shotid)[self.calfib_type]
                if 'comp_all' in locals():
                    comp, var, var_ratio, mean = self.get_pca_one_shot(fib_spec)
                    comp_all = np.append(comp_all, comp[None,:,:], axis=0)
                    var_all = np.append(var_all, var[None,:], axis=0)
                    var_ratio_all = np.append(var_ratio_all, var_ratio[None,:], axis=0)
                    mean_all = np.append(mean_all, mean[None,:], axis=0)
                    shotids_in_pca = np.append(shotids_in_pca, shotid)
                else:
                    comp_all, var_all, var_ratio_all, mean_all = self.get_pca_one_shot(fib_spec)
                    comp_all = comp_all[None,:,:]
                    var_all = var_all[None,:]
                    var_ratio_all = var_ratio_all[None,:]
                    mean_all = mean_all[None,:]
                    shotids_in_pca = np.array([shotid])[None,:]

                if i%10 ==0:
                    self.save_pca(pca_path, comp_all, var_all, var_ratio_all, mean_all, shotids_in_pca)
                progress += 1
            self.save_pca(pca_path, comp_all, var_all, var_ratio_all, mean_all, shotids_in_pca)
        else:
            self.logger.error('Currently only per shot covariance is implemented')
            raise NotImplementedError

        return comp_all, var_all, var_ratio_all, mean_all, shotids_in_pca

    def get_pca_one_shot(self, fib_spec):
        """
        Apply PCA to a given set of fiber spectra.
        Parameters
        ----------
        fib_spec : np.ndarray, shape (N_fibers, N_wavelengths)
            Array containing the fiber spectra.
        Returns
        -------
        components : np.ndarray, shape (N_components, N_wavelengths)
            Principal components.
        explained_variance : np.ndarray, shape (N_components,)
            Explained variance for each principal component.
        explained_variance_ratio : np.ndarray, shape (N_components,)
            Explained variance ratio for each principal component.
        mean_spectrum : np.ndarray, shape (N_wavelengths,)
            Mean spectrum across all fibers.
        """

        if 'l' not in self.cov_options or self.cov_options['l'] is None:
            raise ValueError("The number of PCA components 'l' must be specified in cov_options.")
        n_components = self.cov_options['l']
        # Step 1: Fit PCA
        pca = PCA(n_components=n_components)
        _ = pca.fit_transform(fib_spec)            # shape: (n_fibers, k)
        components = pca.components_                     # shape: (k, m)
        explained_variance = pca.explained_variance_     # shape: (k,)
        explained_variance_ratio = pca.explained_variance_ratio_  # shape: (k,)
        mean_spectrum = pca.mean_                        # shape: (m,)

        self.logger.info(f'Explained variance ratio by top {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}')
        
        return components, explained_variance, explained_variance_ratio, mean_spectrum

    def save_pca(self, pca_path, comp_all, var_all, var_ratio_all, mean_all, shotids_in_pca):
        """
        Save PCA results to disk.
        """
        self.logger.info(f'saving pca in {pca_path}')
        with h5py.File(pca_path, 'w') as fw:
            fw['components'] = comp_all
            fw['explained_variance'] = var_all
            fw['explained_variance_ratio'] = var_ratio_all
            fw['mean_spectrum'] = mean_all
            fw['shotid'] = shotids_in_pca
            fw.attrs['masking'] = json.dumps(self.masking)
            fw.attrs['cov_options'] = json.dumps(self.cov_options)


    def load_pca(self, pca_path):
        """
        Load PCA results from disk.
        """
        if op.exists(pca_path):
            with h5py.File(pca_path, 'r') as f:
                comp_all = f['components'][:]
                var_all = f['explained_variance'][:]
                var_ratio_all = f['explained_variance_ratio'][:]
                mean_all = f['mean_spectrum'][:]
                shotids_in_pca = f['shotid'][:]
            return comp_all, var_all, var_ratio_all, mean_all, shotids_in_pca
        else:
            self.logger.error(f'PCA file {pca_path} does not exist.')
            return None
    
    def get_cov_from_pca(self, pca_path):
        """
        Compute covariance matrices from saved PCA results.
        Parameters
        ----------
        pca_path : str
            Path to the PCA results file.
        Returns
        -------
        cov_matrices : np.ndarray, shape (N_shots, N_wavelengths, N_wavelengths)
            Covariance matrices for each shot computed from PCA results.
        shotids_in_pca : np.ndarray, shape (N_shots,)
            Shot IDs corresponding to the covariance matrices.
        """
        comp_all, var_all, var_ratio_all, mean_all, shotids_in_pca = self.load_pca(pca_path)
        if comp_all is None:
            return None, None
        
        n_shots, n_components, n_wavelengths = comp_all.shape
        cov_matrices = np.zeros((n_shots, n_wavelengths, n_wavelengths))
        
        for i in range(n_shots):
            # Reconstruct covariance matrix from PCA components and explained variance
            cov_matrices[i] = comp_all[i].T @ (var_all[i][:, None] * comp_all[i])
        
        return cov_matrices, shotids_in_pca


class dustin_extra_residual_cleaning():
    def __init__(self):
        pass
    def get_seeing_troughput(self, detect_ids):
        """
        We need to laod the detect object since the 
        cleaned HDR5 cat does not have `fwhm` and `throughput`
        for those shotids
        NOTE: This is slow
        """
        # Loading the detection table is slow
        detects_obj = Detections(loadtable=True)
        self.logger.info(detects_obj.survey)
        detect_table = detects_obj.return_astropy_table()
        ind_detect_table=[]
        # This loops over 1.6e6 sources, so very slow
        for idet in unique_det_id:
            ind = np.where(detect_table['detectid'] == idet)
            ind_detect_table.append(ind)
        det = detects_table[ind_detect_table]['shotid', 'fwhm', 'throughput']
        self.logger.info(f'Total sources {detect_ids.size}, overlap with detect_table {id_detect_table.size}')
        detects.hdfile.close()
        return det
    
    def get_detect_ids():
        """
        Uset latest detected source catalog
        """
        # Get the latest catalog
        cat_file = '/home/jovyan/work/hetdex/data/lae_uniq_5.0.1.fits'
        unique_table = Table.read(cat_file)
        unique_det_id = np.unique(unique_table['detectid'])
        return unique_det_id
    
    def write_useful_info():
        """
        Write `shotid`s, `fwhm` and `throughput` for
        latest LAE catalog.
        """
        unique_det_id = get_detect_ids()
        det = get_seeing_troughput(unique_det_id)
    
        with h5py.File('cat_for_empty_fiber.h5','w') as fw:
            fw['shotid'] =det['shotid']
            fw['fwhm'] = det['fwhm']
            fw['throughput'] = det['throughput']