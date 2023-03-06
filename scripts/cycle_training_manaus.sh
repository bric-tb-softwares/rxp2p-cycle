#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --job-name=cycle_manaus 	# job name
#SBATCH --exclusive         	# Reserve this node only for you
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=cycle_manaus_jobs/Result-%x.%j.out
#SBATCH --error=cycle_manaus_jobs/Result-%x.%j.err
# Runtime of this job

python3 train.py --dataroot /home/otto.tavares/public/iltbi/train/images/ --custom_images_path /home/otto.tavares/public/iltbi/train/images/ --dataset_download_dir /home/otto.tavares/public/iltbi/train/images --download_imgs --model cycle_gan --dataset_mode unalignedskfoldmanaus --token b16fe0fc92088c4840a98160f3848839e68b1148 --project cycle_manaus --name tunning_manaus --preprocess resize_and_scale_width --use_wandb --wandb_fold_id tunning_manaus_v001 --no_flip  --wandb_entity otavares  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 --load_size 256 --crop_size 256 --isTB --test 0 --sort 0
