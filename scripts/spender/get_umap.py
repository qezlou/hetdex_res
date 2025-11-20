import h5py
import numpy as np
import umap


import numpy as np
import importlib

import os.path as op
data_dir = '/scratch/06536/qezlou/hetdex/data/'

modl_file = 'amp_fibs_LL_c149_latents_5_random_sample_combined.pt'
recon_file = 'recon_amp_fibs_LL_random_sample_combined.h5'

reducer = umap.UMAP()

with h5py.File(op.join(data_dir, 'recon', recon_file), 'r') as f:
    latents_train = f['train_latents'][:]
    embeddings = reducer.fit_transform(latents_train)

with h5py.File(op.join(data_dir, 'recon', 'umap'+recon_file[5:]), 'w') as f:
    f.create_dataset('embeddings', data=embeddings)

    