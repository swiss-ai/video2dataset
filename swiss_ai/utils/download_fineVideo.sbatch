#!/bin/bash
#SBATCH --job-name=download-finevideo
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=/store/swissai/a08/containers/v2d/v2d.toml
#SBATCH --account=a08
#SBATCH --reservation=sai-shared


export ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibn


pip install datasets
python /users/julm/a08/julm/video2dataset/swiss_ai/utils/download_fineVideo.py --output_dir "/store/swissai/a08/data/raw/finevideo" --download_all --num_videos_per_tar 10
