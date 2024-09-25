#! /bin/bash
#SBATCH --job-name="BN_dft-d2_157.992_0.0150_7"
#SBATCH --time=4:00:00
#SBATCH --partition="secondary"
#SBATCH --cpus-per-task=20
#SBATCH --output=fit/BN_dft-d2_157.992_0.0150_7/train.out

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
module load anaconda/3
module load openmpi/3.1.1-gcc-7.2.0
. /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate py_env
export ASE_LAMMPSRUN_COMMAND=/home/krongch2/projects/lammps/lammps/src/lmp_mpi
python -u train.py --method dft-d2 --shift 157.992 --dirname BN_dft-d2_157.992_0.0150_7 --basis BN --kT 0.015 --exclude 7
