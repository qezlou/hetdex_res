# I cannot pip install on the Hub
import sys
import os
sys.path.append(os.path.abspath('/home/jovyan/work/hetdex/packs/private-het-data/src/het_cov'))

import fibers

data_dir = '/home/jovyan/work/hetdex/data/'
fibs = fibers.Fibers(data_dir)

fibs.get_cov()