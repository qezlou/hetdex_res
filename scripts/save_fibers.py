"""
We need to save the cleaned fiber spectra (with per-pixel and per-fiber masks applied)
to train on a separate machine than LoneStar6.
"""

import sys
import os
import argparse
import json
import numpy as np
import h5py
from glob import glob

# Determine if we are on the Hub by checking for a specific environment variable
if 'JUPYTERHUB_USER' in os.environ:
    sys.path.append(os.path.abspath('/home/jovyan/work/hetdex/packs/private-het-data/src/het_cov'))
    from het_cov import fibers
    data_dir = '/home/jovyan/work/hetdex/data/'
else:
    from het_cov import fibers
    data_dir = '/scratch/06536/qezlou/hetgen/data'

config = { "masking":{
    "bad_fibers": True,
    "bad_pixels": True,
    "strong_continuum": True,
    "top_varying_pixels": False,
    "top_percent": 5.0,
    "top_fiber_frac": 0.3
    },
    "cov_options": {
        "per": "shot",
        "method": "full",
        "l": 100
    },
    "calfib_type": "calfib",
    "normalize": False       
}


fibs = fibers.Fibers(data_dir,
                    config=config,
                    logging_level='INFO')


fibs.logger.info("Saving spectra on disk")
# This iterates over all date-shots and saves individual shot-spectra
ifuslot, fibnum, multiframe = '013', 10, 'multi_412_013_043_LL'

amp ='LL'

def run_mode(cat='phys-fiber'):

    if cat =='phys-fiber':
        fibs.get_spectra_one_physical_fiber(ifuslot, fibnum, multiframe)
    elif cat == 'amp':
        fibs.get_spectra_one_amp(amp=amp)
    else:
        raise ValueError("cat must be 'phys-fiber' or 'amp'")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute covariance matrices using a JSON config file.")
    parser.add_argument('--cat', type=str, required=True, help="")
    args = parser.parse_args()

    run_mode(cat=args.cat)