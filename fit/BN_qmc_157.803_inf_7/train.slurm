#! /bin/bash
#SBATCH --job-name="BN_qmc_157.803_inf_7"
#SBATCH --time=4:00:00
#SBATCH --partition="secondary"
#SBATCH --cpus-per-task=20
#SBATCH --output=fit/BN_qmc_157.803_inf_7/train.out

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
module load anaconda/3
module load openmpi/3.1.1-gcc-7.2.0
. /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate py_env
export ASE_LAMMPSRUN_COMMAND=/home/krongch2/projects/lammps/lammps/src/lmp_mpi
python -u train.py --method qmc --shift 157.803 --dirname BN_qmc_157.803_inf_7 --basis BN --kT inf --exclude 7
