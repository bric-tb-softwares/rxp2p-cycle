#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --job-name=p2p_santacasa_anonimizado 	# job name
#SBATCH --exclusive         	# Reserve this node only for you
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=Result-%x.%j.out
#SBATCH --error=Result-%x.%j.err
# Runtime of this job

"""
python /app/train.py \
--dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/AB/ \
--dataset_download_dir /home/otto.tavares/public/iltbi/train/images \
--download_imgs \
--model pix2pix \
--dataset_mode santacasaskfoldextendido \
--token {TOKEN} \
--name check_%JOB_NAME \
--preprocess resize_and_scale_width \
--use_wandb \
--wandb_fold_id check_%JOB_NAME \
--wandb_entity brics-tb  \
--gpu_ids %CUDA_VISIBLE_DEVICES \
--n_epochs 500 \
--n_epochs_decay 500 \
--load_size 256 \
--crop_size 256 \
--job %IN \
--project %JOB_TASKNAME \
"""