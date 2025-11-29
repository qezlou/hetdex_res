import os
import numpy as np
from glob import glob

template = """#!/bin/bash
#SBATCH --job-name=cons{i}l10
#SBATCH -A AST25019
#SBATCH -N 1
#SBATCH -p gh
#SBATCH --time=3:00:00

export PYTHONUNBUFFERED=1


# Using one GH node with accelerate
accelerate launch --num_machines 1 train_hetdex.py /scratch/06536/qezlou/hetdex/data/  calfib_consecutive_{i}.h5 /scratch/06536/qezlou/hetdex/data/models/calfib_consecutive_{i}_latents_10.pt -n 10 -b 16384 -f 1.0 -v
"""


fnames = glob('/scratch/06536/qezlou/hetdex/data/fib_spec/calfib_consecutive_*.h5')
all_nums = []
for fn in fnames:
    base = os.path.basename(fn)
    try:
        num = int(base.rsplit('_', 1)[-1].split('.', 1)[0])
    except ValueError:
        continue
    if os.path.isfile(f'/scratch/06536/qezlou/hetdex/data/models/calfib_consecutive_{num}_latents_10.pt'):
        print(f'Skipping existing model for calfib_consecutive_{num}')
        continue
    all_nums.append(num)
all_nums = np.sort(all_nums)
print(f'To submit jobs for files: {all_nums}')
for n in all_nums[1:]:
    print(f'Submitting job for calfib_consecutive_{n}')
    filename = f"job_script_{n}.sh"
    with open(filename, "w") as f:
        f.write(template.format(i=n))
    os.system(f'sbatch job_script_{n}.sh')
    os.remove(f'job_script_{n}.sh')
