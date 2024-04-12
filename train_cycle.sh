#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --job-name=cycle_cmanaus 	# job name
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=Result-%x.%j.out
#SBATCH --error=Result-%x.%j.err
# Runtime of this job
#SBATCH --array=0
python3 train.py --dataroot /home/brics/public/brics_data/Manaus/c_manaus/ --dataset_download_dir /home/otto.tavares/public/iltbi/train/images --model cycle_gan --dataset_mode unalignedskfoldcmanaus --token b16fe0fc92088c4840a98160f3848839e68b1148 --project tunning_cycle_cmanaus --name tunning_cycle_cmanaus --preprocess resize_and_scale_width --use_wandb --wandb_fold_id tunning_cmanaus_v001 --no_flip  --wandb_entity otavares  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 --load_size 256 --crop_size 256 --isTB --test 0 --sort $SLURM_ARRAY_TASK_ID





