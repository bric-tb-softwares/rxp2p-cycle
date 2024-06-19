

export MODELS='/home/otto.tavares/public/iltbi/rxp2p-cycle/production/manaus/c_manaus/p2p/user.otto.tavares.Manaus.c_manaus.pix2pix_v1.notb.r1/'
#export MODELS='/home/otto.tavares/public/iltbi/rxp2p-cycle/production/manaus/c_manaus/p2p/user.otto.tavares.Manaus.c_manaus.pix2pix_v1.tb.r1/'

#NTB
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode cmanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/c_manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_train_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode cmanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/c_manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_val_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode cmanausskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/c_manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_test_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images

#TB
#python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode cmanausskfold2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/c_manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_train_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
#python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode cmanausskfold2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/c_manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_val_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
#python ./generate.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode cmanausskfold2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/c_manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_test_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images

mkdir user.otto.tavares.Manaus.c_manaus.pix2pix_v1.notb.r1.samples
mv fake_images_ntb/results user.otto.tavares.Manaus.c_manaus.pix2pix_v1.notb.r1.samples
rm -rf fake_images_ntb


#mkdir user.otto.tavares.Manaus.c_manaus.pix2pix_v1.tb.r1.samples
#mv fake_images_tb/results user.otto.tavares.Manaus.c_manaus.pix2pix_v1.tb.r1.samples
#rm -rf fake_images_tb
