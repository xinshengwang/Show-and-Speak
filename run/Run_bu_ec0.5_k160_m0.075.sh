#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=65:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:turing:1
##SBATCH --nodelist=awi02
#SBATCH --exclude=insy6,influ1,influ3
#SBATCH --chdir=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/code/Image2speech/Proposed_method/BU_with_info_with_ec_schedule_sampling_with_max

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.5.1.10
srun -u --output=run/bu_iec0.5_k160_m0.075.outputs sh run/bu_iec0.5_k160_m0.075.sh