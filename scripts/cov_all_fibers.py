# I cannot pip install on the Hub
import sys
import os
import importlib
import json
import numpy as np
from het_cov import fibers
importlib.reload(fibers)

# Determine if we are on the Hub by checking for a specific environment variable
if 'JUPYTERHUB_USER' in os.environ:
    data_dir = '/home/jovyan/work/hetdex/data/'
else:
    data_dir = '/work/06536/qezlou/hetdex/data/'

config={ "masking":{
    "bad_fibers": True,
    "bad_pixels": True,
    "strong_continuum": True,
    "top_varying_pixels": False,
    "top_percent": 5.0,
    "top_fiber_frac":0.3
    },
    "cov_options": {
        "per": "shot",
        "method": "full",
        "l": 100
    },
    "calfib_type": "calfib",
    "normalize": False       
}

fibs = fibers.Fibers(data_dir, config=config)




import numpy as np
import multiprocessing as mp

# Define worker function
def process_shot(i):
    shotid = fibs.shotids_list[i]
    fibs_tab = fibs.get_fibers_one_shot(shotid, keep_calfibe=True)
    return fibs_tab['calfib'][:], fibs_tab['calfibe'][:]


indices = np.arange(-1, -30, -1)

with mp.Pool(processes=mp.cpu_count()) as pool:
    results = pool.map(process_shot, indices)

# Separate calfibs and calfibse, stacking after
calfibs_list, calfibse_list = zip(*results)
calfibs = np.vstack(calfibs_list)
calfibse = np.vstack(calfibse_list)

cov = np.cov(calfibs, rowvar=False)

# Save to h5file
import h5py
with h5py.File('cov.5', 'w') as hf:
    hf.create_dataset('/work/06536/qezlou/hetdex/data/cov_raw_30_fibers.h5', data=cov)
    hf.create_dataset('mean_calfibse', data=np.mean(calfibse))