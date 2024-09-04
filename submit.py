import os
import numpy as np

import train

def get_hours(queue):
    hours_map = {'secondary': 4, 'qmchamm': 4, 'physics': 14, 'test': 1}
    return hours_map[queue]

def submit_local(cmd):
    os.system(cmd)

def submit_slurm(dirname, cmd, queue='secondary'):
    submit_filename = f'fit/{dirname}/train.slurm'
    hours = get_hours(queue)

    with open(submit_filename, 'w') as f:
        f.write(
f'''#! /bin/bash
#SBATCH --job-name="{dirname}"
#SBATCH --time={hours}:00:00
#SBATCH --partition="{queue}"
#SBATCH --cpus-per-task=20
#SBATCH --output=fit/{dirname}/train.out

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
module load anaconda/3
module load openmpi/3.1.1-gcc-7.2.0
. /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate py_env
export ASE_LAMMPSRUN_COMMAND=/home/krongch2/projects/lammps/lammps/src/lmp_mpi
{cmd}
''')
    os.system(f'sbatch {submit_filename}')

def submit_batch():
    os.makedirs('fit', exist_ok=True)
    for basis in ['C2', 'BN']:
        for method in ['dft-d3', 'dft-d2']:
            middle_shift = train.find_shift(method, basis)
            print('middle_shift', middle_shift)
            # for extra_shift in np.arange(-0.005, 0.005+0.001, 0.001):
            for extra_shift in [0]:
                shift = middle_shift + extra_shift
                print('total_shift', shift)
                # for kT in np.arange(0.001, 0.020+0.001, 0.001):
                for kT in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.0060, 0.0080, 0.0090, 0.0100, 0.0150, 0.0200, 'inf']:
                    exclude = '567'
                    # exclude = '7'
                    print(kT)
                    kT_str = 'inf' if kT == 'inf' else f'{kT:.4f}'
                    dirname = f'{basis}_{method}_{shift:.3f}_{kT_str}_{exclude}'
                    cmd = f'python -u train.py --method {method} --shift {shift} --dirname {dirname} --basis {basis} --kT {kT} --exclude {exclude}'
                    os.makedirs(f'fit/{dirname}', exist_ok=True)
                    submit_slurm(dirname, cmd)
                    # submit_local(cmd)

if __name__ == '__main__':
    submit_batch()
