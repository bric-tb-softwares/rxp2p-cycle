#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --exclude=caloba78      #excluir gpu-large caloba91 e gpu caloba78
#SBATCH --job-name=cycle_ntb_santacasa 	# job name
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=logs/Result-%x.%j.out
#SBATCH --error=logs/Result-%x.%j.err
# Runtime of this job
#SBATCH --array=2
python3 train.py --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid --dataset_download_dir /home/otto.tavares/public/iltbi/train/images --model cycle_gan --dataset_mode unalignedskfoldextendido --token b16fe0fc92088c4840a98160f3848839e68b1148 --project cycle_shenzhen_ntb_santacasa --name test_1_sort_$SLURM_ARRAY_TASK_ID --preprocess resize_and_scale_width --use_wandb --wandb_fold_id fixV1_cycle_ntb_test_1_sort_$SLURM_ARRAY_TASK_ID --no_flip  --wandb_entity otavares  --gpu_ids 0 --n_epochs 100 --n_epochs_decay 100 --load_size 256 --crop_size 256 --save_epoch_freq 50 --test 1 --sort $SLURM_ARRAY_TASK_ID





