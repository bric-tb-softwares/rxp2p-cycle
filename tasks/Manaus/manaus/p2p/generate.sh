

#export MODELS='/home/brics/public/brics_data/Manaus/manaus/models/user.otto.tavares.Manaus.manaus.pix2pix_v1.notb.r3/'
export MODELS='/home/brics/public/brics_data/Manaus/manaus/models/user.otto.tavares.Manaus.manaus.pix2pix_v1.tb.r3/'

#NTB
#python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode manausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_train_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
#python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode manausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_val_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
#python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode manausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_test_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images

#TB
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode manausskfold2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_train_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode manausskfold2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_val_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode manausskfold2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_test_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images

#mkdir user.otto.tavares.Manaus.manaus.pix2pix_v1.notb.r3.samples
#mv fake_images_ntb/results user.otto.tavares.Manaus.manaus.pix2pix_v1.notb.r3.samples
#rm -rf fake_images_ntb


mkdir user.otto.tavares.Manaus.manaus.pix2pix_v1.tb.r3.samples
mv fake_images_tb/results user.otto.tavares.Manaus.manaus.pix2pix_v1.tb.r3.samples
rm -rf fake_images_tb
