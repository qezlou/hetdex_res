import sys
import os
import argparse
import json

# Determine if we are on the Hub by checking for a specific environment variable
if 'JUPYTERHUB_USER' in os.environ:
    sys.path.append(os.path.abspath('/home/jovyan/work/hetdex/packs/private-het-data/src/het_cov'))
    from het_cov import fibers
    data_dir = '/home/jovyan/work/hetdex/data/'
else:
    from het_cov import fibers
    data_dir = '/work/06536/qezlou/hetdex/data/'

def run(config, save_file):
    # Load masking options from config or use defaults
    masking = config.get('masking', {
        'bad_fibers': True,
        'bad_pixels': True,
        'strong_continuum': True
    })

    # Load covariance options from config or use defaults
    cov_options = config.get('cov_options', {
        'per': 'shot',
        'method': 'pca',
        'l': 20
    })

    fibs = fibers.Fibers(data_dir, 
                         masking=masking,
                         cov_options=cov_options,
                         logging_level='INFO')


    fibs.get_cov(save_file=save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute covariance matrices using a JSON config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file with options.")
    args = parser.parse_args()

    # Determine output file name based on the json config file name
    save_file = os.path.basename(args.config)
    if save_file.endswith('.json'):
        save_file = save_file[:-5] + '.h5'
    else:
        save_file = save_file + '.h5'

    with open(args.config, 'r') as f:
        config = json.load(f)

    run(config, save_file=save_file)