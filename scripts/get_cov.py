# I cannot pip install on the Hub
import sys
import os
import argparse

# Determine if we are on the Hub by checking for a specific environment variable
if 'JUPYTERHUB_USER' in os.environ:
    sys.path.append(os.path.abspath('/home/jovyan/work/hetdex/packs/private-het-data/src/het_cov'))
    from het_cov import fibers
    data_dir = '/home/jovyan/work/hetdex/data/'
else:
    from het_cov import fibers
    data_dir = '/work/06536/qezlou/hetdex/data/'

def run(bad_fibs=True, bad_pix=True, strong_cont=True):
    fibs = fibers.Fibers(data_dir, logging_level='INFO')


    masking={'bad_fibers': bad_fibs, 'bad_pixels': bad_pix, 'strong_continuum': strong_cont}

    if strong_cont:
        save_File = 'cov_calfib_ffsky_rmvd_bad_fibs_cont.h5'
    elif bad_fibs:
        save_File = 'cov_calfib_ffsky_rmvd_bad_fibs.h5'
    
    fibs.get_cov(save_file=save_File, masking=masking)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute covariance matrices with optional masking.")
    parser.add_argument('--bad_fibs', type=int, help="Mask bad fibers if set.")
    parser.add_argument('--bad_pix', type=int, help="Mask bad pixels if set.")
    parser.add_argument('--strong_cont', type=int, help="Mask strong continuum if set.")
    args = parser.parse_args()

    run(bad_fibs=args.bad_fibs, bad_pix=args.bad_pix, strong_cont=args.strong_cont)