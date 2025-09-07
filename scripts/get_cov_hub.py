# I cannot pip install on the Hub
import sys
import os

# Determine if we are on the Hub by checking for a specific environment variable
if 'JUPYTERHUB_USER' in os.environ:
    sys.path.append(os.path.abspath('/home/jovyan/work/hetdex/packs/private-het-data/src/het_cov'))
    import fibers
    data_dir = '/home/jovyan/work/hetdex/data/'
else:
    import het_cov import fibers
    data_dir = '/work/06536/qezlou/hetdex/data/'


fibs = fibers.Fibers(data_dir)

fibs.get_cov()