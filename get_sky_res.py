"""
Mostly based on what Mahan is doing for absorption
"""

from astropy.table import Table, vstack, hstack, join
from hetdex_api.survey import Survey, FiberIndex
from hetdex_api.shot import Fibers
from elixer import spectrum_utilities as SU
from hetdex_api.detections import Detections
import numpy as np
import h5py

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

        

    
    
