#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu-large      	# The partion name: gpu or cpu
#SBATCH --exclude=caloba78      #excluir gpu-large caloba91 e gpu caloba78
#SBATCH --job-name=gen_cycle 	# job name
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=logs/Result-%x.%j.out
#SBATCH --error=logs/Result-%x.%j.err
#python generate_synth.py --checkpoints_dir /home/otto.tavares/public/iltbi/rxpixp2pixcycle/checkpoints/  --results_dir /home/otto.tavares/public/iltbi/rxpixp2pixcycle/versions/v1/user.otavares.cycle.shenzhenSantaCasa.v1_BtoA.r1/1aPRODUCAO/  --dataroot /home/brics/public/brics_data/Shenzhen --dataset_download_dir /home/otto.tavares/public/iltbi/train/images --download_imgs --custom_images_path /home/brics/public/brics_data/Shenzhen/raw/images --model test2cycle --direction BtoA --netG resnet_9blocks  --dataset_mode cyclesantacasaskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --n_folds 10 --preprocess resize_and_scale_width --no_dropout --gpu_ids 0 --name genkl --isTB --gen_test #--gen_train_dataset

export MODELS='/home/otto.tavares/public/iltbi/rxp2p-cycle/production/santa_casa/imageamento_anonimizado_valid/cycle/user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.cycle_notb_v1.r5'
export TARGET='/home/otto.tavares/public/iltbi/rxp2p-cycle/production/santa_casa/imageamento_anonimizado_valid/cycle'
export ROOT='/home/otto.tavares/public/iltbi/rxp2p-cycle'
#A to B - Entra Santa Casa e sai Shenzhen NTB
python $ROOT/generate.py --checkpoints_dir $MODELS --results_dir $TARGET/fake_images_notb_SantaCasa_Shenzhen --gpu_ids 0 --dataset_mode  cyclesantacasaskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction AtoB --netG resnet_9blocks --gen_train_dataset --n_folds 10 --no_dropout --name genkl --dataset_download_dir $PWD/images
python $ROOT/generate.py --checkpoints_dir $MODELS --results_dir $TARGET/fake_images_notb_SantaCasa_Shenzhen --gpu_ids 0 --dataset_mode  cyclesantacasaskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction AtoB --netG resnet_9blocks --gen_val_dataset --n_folds 10 --no_dropout --name genkl --dataset_download_dir $PWD/images
python $ROOT/generate.py --checkpoints_dir $MODELS --results_dir $TARGET/fake_images_notb_SantaCasa_Shenzhen --gpu_ids 0 --dataset_mode  cyclesantacasaskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction AtoB --netG resnet_9blocks --gen_test_dataset --n_folds 10 --no_dropout --name genkl --dataset_download_dir $PWD/images


#B to A - Entra Shenzhen NTB e sai Santa Casa 
#python ./generate.py --checkpoints_dir $MODELS --results_dir fake_images_notb_Shenzhen_SantaCasa --gpu_ids 0 --dataset_mode  cyclesantacasaskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction BtoA --netG resnet_9blocks --gen_train_dataset --n_folds 10 --no_dropout --name genkl --dataset_download_dir $PWD/images
#python ./generate.py --checkpoints_dir $MODELS --results_dir fake_images_notb_Shenzhen_SantaCasa --gpu_ids 0 --dataset_mode  cyclesantacasaskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction BtoA --netG resnet_9blocks --gen_val_dataset --n_folds 10 --no_dropout --name genkl --dataset_download_dir $PWD/images
#python ./generate.py --checkpoints_dir $MODELS --results_dir fake_images_notb_Shenzhen_SantaCasa --gpu_ids 0 --dataset_mode  cyclesantacasaskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction BtoA --netG resnet_9blocks --gen_test_dataset --n_folds 10 --no_dropout --name genkl --dataset_download_dir $PWD/images

#
#mv fake_images_SantaCasa_Shenzhen/results .
#mv results user.otto.tavares.SantaCasa_imageamento_anonimizado_valid_Shenzehn.cycle_v1.r1
#rm -rf fake_images_SantaCasa_Shenzhen
#
#mv fake_images_Shenzhen_SantaCasa/results .
#mv results user.otto.tavares.Shenzhen_SantaCasa_imageamento_anonimizado_valid.cycle_v1.r1
#rm -rf fake_images_Shenzhen_SantaCasa
#