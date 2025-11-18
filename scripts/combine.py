import numpy as np
import h5py
from glob import glob
import os
import sys
import random


data_dir = '/scratch/06536/qezlou/hetdex/data/fib_spec/' 
files = glob(os.path.join(data_dir, 'amp_fibs_LL_c*.h5'))
random.shuffle(files)
shotids = []
calfibs_list = []
calfibse_list = []
expnums= []
c=0
print(f'Combining {len(files)} files...')
for i, file in enumerate(files):
    print(f'Processing file {i+1}/{len(files)}: {file}')
    with h5py.File(file, 'r') as fr:
        calfibs_list.append(fr['calfib'][:])
        calfibse_list.append(fr['calfibe'][:])
        shotids.extend(fr['shotids'][:])
        expnums.extend(fr['expnum'][:])
    if ((i+1)%3 == 0 and i>0) or (i == len(files)-1):
        # Save all on one file
        print(f'saving chunk {c} with {len(shotids)} entries, min-max dateshot {min(shotids)}-{max(shotids)}')
        with h5py.File(os.path.join(data_dir, f'amp_fibs_LL_shuffled{c}.h5'), 'w') as fw:
            fw['shotids'] = shotids
            fw['expnums'] = expnums
            fw['calfib'] = np.vstack(calfibs_list)
            fw['calfibe'] = np.vstack(calfibse_list)
        shotids = []
        calfibs_list = []
        calfibse_list = []
        expnums= []
        c+=1
