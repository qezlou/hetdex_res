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

    fibs = fibers.Fibers(data_dir, 
                         config = config,
                         logging_level='INFO')

    if fibs.cov_options['method'] == 'pca':
        fibs.logger.info("Running PCA method")
        fibs.do_pca()
    elif fibs.cov_options['method'] == 'full':
        fibs.logger.info("Running full covariance method")
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


"""
Example usage:
python get_cov.py --config scripts/cov_pca.json
Te config file (scripts/cov_pca.json) should look like:
{
    "masking": {
        "bad_fibers": true,
        "bad_pixels": true,
        "strong_continuum": true
    },
    "cov_options": {
        "per": "shot",
        "method": "pca",
        "l": 50
    }
}
"""