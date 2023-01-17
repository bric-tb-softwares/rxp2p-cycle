#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu-large      	# The partion name: gpu or cpu
#SBATCH --job-name=gen_p2p_santacasa 	# job name
#SBATCH --exclusive         	# Reserve this node only for you
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=output_jobs_train_p2p/gen_jobs_p2p/Result-%x.%j.out
#SBATCH --error=output_jobs_train_p2p/gen_jobs_p2p/Result-%x.%j.err
# Runtime of this job

python generate_synth.py --checkpoints_dir /home/otto.tavares/public/iltbi/rxpixp2pixcycle/checkpoints/  --results_dir /home/otto.tavares/public/iltbi/rxpixp2pixcycle/versions/v1/user.otavares.SantaCasa.pix2pix.v1_iltbi.r1/1aPRODUCAO/  --dataroot /home/otto.tavares/public/iltbi/train/images/santa_casa_pares/ --custom_images_path /home/otto.tavares/public/iltbi/train/images/santa_casa_pares/  --dataset_mode santacasaskfold2generator --model test  --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_test_dataset --n_folds 10 --name genkl 