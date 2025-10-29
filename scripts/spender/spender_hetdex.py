import os
from spender.data import hetdex

# Test data laoding

data_dir = '/scratch/06536/qezlou/hetdex/data/'
outfile = '/scratch/06536/qezlou/hetdex/data/models/'
hetdex_data = hetdex.HETDEX()

data_loader = hetdex_data.get_data_loader(data_dir, which='train')
print(f'Number of batches: {len(data_loader)} for training data')

data_loader = hetdex_data.get_data_loader(data_dir, which='valid')
print(f'Number of batches: {len(data_loader)} for validation data')

# Test quick training run

os.system(f'accelerate launch --num_processes 1 train_hetdex.py {data_dir}  {outfile} -v')