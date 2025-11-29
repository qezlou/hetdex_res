from recon_spec import Reconstructor
import os
import h5py
from glob import glob

data_dir = '/scratch/06536/qezlou/hetdex/data/'


def recon_one_bundle(spec_file, model_file, output_file):
        

    reconstructor = Reconstructor(
        data_dir=data_dir,
        spec_file=spec_file,
        model_file=model_file,
        wave_obs=None,
        which='both'
    )
    latents, recon, inds = reconstructor.reconstruct_spectra()
    out_path = os.path.join(data_dir, 'recon', output_file)
    with h5py.File(out_path, 'w') as f:
            f.create_dataset('latents', data=latents)
            f.create_dataset('recon_spectra', data=recon)
            f.create_dataset('inds', data=inds)


def find_bundle_num(fname):
    base = os.path.basename(fname)
    try:
        num = int(base.rsplit('_')[2])
    except ValueError:
        return None
    return num

def find_remaining_bundles(data_dir):
    fnames = glob(os.path.join(data_dir, 'models', 'calfib_consecutive_*_latents_10.pt'))
    all_nums = []
    for fn in fnames:
        num = find_bundle_num(fn)
        if num is not None:
            if not os.path.isfile(os.path.join(data_dir, 'recon', f'recon_calfib_consecutive_{num}_latents_10.h5')):
                all_nums.append(num)
    return sorted(all_nums)


all_nums = find_remaining_bundles(data_dir)
print(f'remaining bundles: len={len(all_nums)} nums={all_nums}')

for bundle_num in all_nums:
    spec_file = f'calfib_consecutive_{bundle_num}.h5'
    model_file = f'calfib_consecutive_{bundle_num}_latents_10.pt'
    output_file = f'recon_calfib_consecutive_{bundle_num}_latents_10.h5'
    print(f'Reconstructing bundle {bundle_num}...')
    recon_one_bundle(spec_file, model_file, output_file)
    print(f'Finished bundle {bundle_num}.')