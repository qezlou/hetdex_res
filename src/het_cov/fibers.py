"""
Mostly based on what Mahan is doing for absorption
"""

from astropy.table import Table, join
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import FiberIndex
from hetdex_api.detections import Detections
import numpy as np
import h5py
import os.path as op

class Fibers():
    """
    Getting all "good fibers" spectra for all shots in HDR5
    Should run on both JupyterHub and Compute nodes on Lonestar6
    """
    def __init__(self, data_dir= '/work2/06536/qezlou/hetdex/data/'):
        """
        Parameters
        ----------
        data_dir: str
            Directory where the h5 file with shotid list is located
        """
        self.data_dir = data_dir
        self.save_dir = data_dir
        self.shotids_list = self.load_shotids_list(data_dir)

    def _load_shotids_list(self, data_dir,):
        """
        Load the shotid list from a h5 file that was generated with 
        `hetdex_api.detections.Detections`
        """
        if op.exists(op.join(data_dir, 'hdr5_shotid_list.h5')):
            with h5py.File(op.join(data_dir, 'hdr5_shotid_list.h5'),'r') as f:
                shotids_list = f['shotid'][:]
        else:
            print('Shotid list not found, generating it now')
            shotids_list = self.load_shotids_list()
        return shotids_list

    def load_shotids_list(self):
        """
        Load the shotid list from a h5 file that was generated with 
        `hetdex_api.detections.Detections`
        """
        # Get shot_ids for HDR5
        # NOTE: This is a bit slow
        detects_obj = Detections(loadtable=True)
        print(detects_obj.survey)
        detect_table = detects_obj.return_astropy_table()
        del detects_obj
        shotids_list = np.unique(detect_table['shotid'])
        del detect_table
        print(f'we have {len(shotids_list)} shotids')
        with h5py.File(op.join(self.data_dir, 'hdr5_shotid_list.h5'),'w') as fw:
            fw['shotid'] = shotids_list

    def _get_fibers_one_shot(shotid):
        """
        Get fiber table for a single shot
        Parameters
        ----------
        shotid: int
            Shot ID
        Returns
        -------
        fib_tab: astropy Table
            Table with fiber_id, calfib_ffsky and flag (True for good fibers)
        """
        fibtable_one_shot = get_fibers_table(shot=shotid, survey='hdr5',verbose=False, add_rescor=False)['fiber_id','calfib_ffsky'] # I won't use  'calfibe' 
        F = FiberIndex(survey='hdr5') 
        fib_tab_findex = F.return_shot( shotid)['fiber_id','flag']
        fib_tab= join(fibtable_one_shot, fib_tab_findex, "fiber_id" )
        # Only keep good fibers, flag=True
        print(f'Total fibers: {len(fib_tab)}')
        fib_tab = fib_tab[fib_tab['flag']==True]
        print(f'Good fibers: {len(fib_tab)}')
        return fib_tab

    def get_fibers(self):
        """
        Iterate over all shotids and save the `calfib_ffsky` spectra
        for each shotid in a separate h5 file.
        """
        for shotid in self.shotids_list:
            fib_tab = self._get_fibers_one_shot(shotid)
            with h5py.File(op.join(f'calfib_ffsky_{shotid}.h5'), 'w') as fw:
                fw['calfib_ffsky'] = fib_tab['calfib_ffsky']
            
    def get_covariance(self):
        """
        Iterate over all shotids and compute the covariance matrix
        for the `calfib_ffsky` spectra and save it in a separate h5 file.
        """
        for shotid in self.shotids_list:
            fib_spec = self._get_fibers_one_shot(shotid)['calfib_ffsky']
            cov = np.cov(fib_spec, rowvar=False)
            with h5py.File(op.join(self.save_dir, f'cov_calfib_ffsky_{shotid}.h5'), 'w') as fw:
                fw['cov_calfib_ffsky'] = cov

class dustin_extra_residual_cleaning():
    def __init__():
        pass
    def get_seeing_troughput(detect_ids):
        """
        We need to laod the detect object since the 
        cleaned HDR5 cat does not have `fwhm` and `throughput`
        for those shotids
        NOTE: This is slow
        """
        # Loading the detection table is slow
        detects_obj = Detections(loadtable=True)
        print(detects_obj.survey)
        detect_table = detects_obj.return_astropy_table()
        ind_detect_table=[]
        # This loops over 1.6e6 sources, so very slow
        for idet in unique_det_id:
            ind = np.where(detect_table['detectid'] == idet)
            ind_detect_table.append(ind)
        det = detects_table[ind_detect_table]['shotid', 'fwhm', 'throughput']
        print(f'Total sources {detect_ids.size}, overlap with detect_table {id_detect_table.size}')
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

        

    




class dustin_extra_residual_cleaning():
    def __init__():
        pass
    def get_seeing_troughput(detect_ids):
        """
        We need to laod the detect object since the 
        cleaned HDR5 cat does not have `fwhm` and `throughput`
        for those shotids
        NOTE: This is slow
        """
        # Loading the detection table is slow
        detects_obj = Detections(loadtable=True)
        print(detects_obj.survey)
        detect_table = detects_obj.return_astropy_table()
        ind_detect_table=[]
        # This loops over 1.6e6 sources, so very slow
        for idet in unique_det_id:
            ind = np.where(detect_table['detectid'] == idet)
            ind_detect_table.append(ind)
        det = detects_table[ind_detect_table]['shotid', 'fwhm', 'throughput']
        print(f'Total sources {detect_ids.size}, overlap with detect_table {id_detect_table.size}')
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

        

    