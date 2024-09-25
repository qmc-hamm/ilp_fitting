#! /bin/bash
#SBATCH --job-name="C2_dft-d3_147.474_0.0025_567"
#SBATCH --time=4:00:00
#SBATCH --partition="secondary"
#SBATCH --cpus-per-task=20
#SBATCH --output=fit/C2_dft-d3_147.474_0.0025_567/train.out

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
module load anaconda/3
module load openmpi/3.1.1-gcc-7.2.0
. /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate py_env
export ASE_LAMMPSRUN_COMMAND=/home/krongch2/projects/lammps/lammps/src/lmp_mpi
python -u train.py --method dft-d3 --shift 147.474 --dirname C2_dft-d3_147.474_0.0025_567 --basis C2 --kT 0.0025 --exclude 567
