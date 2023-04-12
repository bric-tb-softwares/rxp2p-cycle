#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --job-name=p2p_manaus 	# job name
#SBATCH --exclusive         	# Reserve this node only for you
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=Result-%x.%j.out
#SBATCH --error=Result-%x.%j.err
# Runtime of this job

python train.py \
--dataroot /home/otto.tavares/public/iltbi/train/images/Manaus_AB/ \
--dataset_download_dir /home/otto.tavares/public/iltbi/train/images \
--model pix2pix \
--dataset_mode manausskfold \
--token b16fe0fc92088c4840a98160f3848839e68b1148 \
--name manaus_tunning_0004 \
--preprocess resize_and_scale_width \
--use_wandb \
--wandb_fold_id manaus_p2p_tunning_004 \
--wandb_entity brics-tb \
--gpu_ids 0 \
--n_epochs 500 \
--n_epochs_decay 500 \
--load_size 256 \
--crop_size 256 \
--project tunning_p2p_manaus \
--save_epoch_freq 200 \
--isTB