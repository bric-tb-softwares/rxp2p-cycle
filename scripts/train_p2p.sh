#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --job-name=p2p_santacasa 	# job name
#SBATCH --exclusive         	# Reserve this node only for you
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=Result-%x.%j.out
#SBATCH --error=Result-%x.%j.err
# Runtime of this job
#SBATCH --array=0-9
python3 train.py --dataroot /home/otto.tavares/public/iltbi/train/images/santa_casa_pares/ --custom_images_path /home/otto.tavares/public/iltbi/train/images/santa_casa_pares/ --model pix2pix --dataset_mode santacasaskfold  --project p2p_santacasa --name santacasa_test_9_sort_$SLURM_ARRAY_TASK_ID --preprocess resize_and_scale_width --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 --load_size 256 --crop_size 256 --val_freq 10  --lambda_L1 50 --gan_mode lsgan --test 9 --sort $SLURM_ARRAY_TASK_ID --use_wandb --wandb_entity otavares --wandb_fold_id def_off_wandb_test_9_sort_$SLURM_ARRAY_TASK_ID





