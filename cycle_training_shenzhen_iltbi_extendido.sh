#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --job-name=cycle_shenzhen_santacasa_anonimizado 	# job name
#SBATCH --exclusive         	# Reserve this node only for you
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=cycle_shenzhen_iltbi_extendido_jobs/Result-%x.%j.out
#SBATCH --error=cycle_shenzhen_iltbi_extendido_jobs/Result-%x.%j.err
# Runtime of this job

python train.py --dataroot /home/brics/public/brics_data/Shenzhen --dataset_download_dir /home/otto.tavares/public/iltbi/train/images --download_imgs --custom_images_path /home/brics/public/brics_data/Shenzhen/raw/images --model cycle_gan --dataset_mode unalignedskfoldextendido --token ... --project cycle_gan_shenzhen_imageamento_extendido --name check_test_sort_00 --preprocess resize_and_scale_width --use_wandb --wandb_fold_id check_test_sort_00_v0 --no_flip  --wandb_entity otavares  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 --load_size 256 --crop_size 256 --isTB --test 0 --sort 0
