
export MODELS='/home/brics/public/brics_data/Manaus/manaus/models/user.otto.tavares.Manaus.manaus.cycle_v1.r2/'


#A to B - Entra Santa Casa e sai Manaus TB
#python ./generate.py --checkpoints_dir $MODELS --results_dir fake_images_SantaCasa_Manaus --gpu_ids 0 --dataset_mode  cyclemanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction AtoB --netG resnet_9blocks --gen_train_dataset --n_folds 10 --no_dropout --name genkl --isTB --dataset_download_dir $PWD/images
#python ./generate.py --checkpoints_dir $MODELS --results_dir fake_images_SantaCasa_Manaus --gpu_ids 0 --dataset_mode  cyclemanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction AtoB --netG resnet_9blocks --gen_val_dataset --n_folds 10 --no_dropout --name genkl --isTB --dataset_download_dir $PWD/images
#python ./generate.py --checkpoints_dir $MODELS --results_dir fake_images_SantaCasa_Manaus --gpu_ids 0 --dataset_mode  cyclemanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/images/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --norm instance --direction AtoB --netG resnet_9blocks --gen_test_dataset --n_folds 10 --no_dropout --name genkl --isTB --dataset_download_dir $PWD/images


#
##B to A - Entra Manaus TB sai Santa Casa
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_Manaus_SantaCasa --dataset_mode  cyclemanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --direction BtoA --netG resnet_9blocks --norm instance  --gpu_ids 0 --gen_train_dataset --n_folds 10 --no_dropout --name genkl --isTB --dataset_download_dir $PWD/images
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_Manaus_SantaCasa --dataset_mode  cyclemanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --direction BtoA --netG resnet_9blocks --norm instance  --gpu_ids 0 --gen_val_dataset --n_folds 10 --no_dropout --name genkl --isTB --dataset_download_dir $PWD/images
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_Manaus_SantaCasa --dataset_mode  cyclemanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/ --model test2cycle --n_folds 10 --preprocess resize_and_scale_width  --direction BtoA --netG resnet_9blocks --norm instance  --gpu_ids 0 --gen_test_dataset --n_folds 10 --no_dropout --name genkl --isTB --dataset_download_dir $PWD/images
§



