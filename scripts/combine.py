import numpy as np
import h5py
from glob import glob
import os
import sys

# Determine if we are on the Hub by checking for a specific environment variable
if 'JUPYTERHUB_USER' in os.environ:
    sys.path.append(os.path.abspath('/home/jovyan/work/hetdex/packs/private-het-data/src/het_cov'))
    from het_cov import fibers
    data_dir = '/home/jovyan/work/hetdex/data/'
else:
    from het_cov import fibers
    data_dir = '/scratch/06536/qezlou/hetgen/data/'

files = glob(os.path.join(data_dir, 'calfib_*.h5'))
shotids = []
calfibs_list = []
calfibse_list = []
out_file = os.path.join(data_dir, 'calfibs.h5')
c=0
print(f'Combining {len(files)} files...')
for i, file in enumerate(files):
    print(f'Processing file {i+1}/{len(files)}: {file}')
    with h5py.File(file, 'r') as fr:
        calfibs_list.append(fr['calfib'][:])
        calfibse_list.append(fr['calfibe'][:])
        shotids.append(fr['shotid'][()])
    if (i%20 == 0 and i>0) or (i == len(files)-1):
        # Save all on one file
        all_shotids = np.repeat(shotids, [cf.shape[0] for cf in calfibs_list])
        with h5py.File(out_file[:-3]+f'_set2_{c}.h5', 'w') as fw:
            fw['shotids'] = all_shotids
            fw['calfib'] = np.vstack(calfibs_list)
            fw['calfibe'] = np.vstack(calfibse_list)
        calfibs_list = []
        calfibse_list = []
        shotids = []
        c+=1