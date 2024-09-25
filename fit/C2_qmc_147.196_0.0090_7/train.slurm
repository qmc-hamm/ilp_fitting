#! /bin/bash
#SBATCH --job-name="C2_qmc_147.196_0.0090_7"
#SBATCH --time=4:00:00
#SBATCH --partition="secondary"
#SBATCH --cpus-per-task=20
#SBATCH --output=fit/C2_qmc_147.196_0.0090_7/train.out

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
module load anaconda/3
module load openmpi/3.1.1-gcc-7.2.0
. /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate py_env
export ASE_LAMMPSRUN_COMMAND=/home/krongch2/projects/lammps/lammps/src/lmp_mpi
python -u train.py --method qmc --shift 147.196 --dirname C2_qmc_147.196_0.0090_7 --basis C2 --kT 0.009 --exclude 7
