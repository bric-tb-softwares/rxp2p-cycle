

export MODELS='/home/brics/public/brics_data/Manaus/manaus/models/user.otto.tavares.Manaus_manaus.pix2pix_v1.ntb.r3/'
#export MODELS='/home/brics/public/brics_data/Manaus/manaus/models/user.otto.tavares.Manaus_manaus.pix2pix_v1.tb.r3/'

#NTB
python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode manausskfoldextendido2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_train_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode manausskfoldextendido2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_val_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images_ntb --dataset_mode manausskfoldextendido2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_test_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images

#TB
#python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode manausskfoldextendido2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_train_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
#python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode manausskfoldextendido2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_val_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
#python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images_tb --dataset_mode manausskfoldextendido2generator --isTB --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/Manaus/manaus/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_test_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images


mv fake_images_ntb/results .
mv results user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.pix2pix_v1.r3.ntb.samples
rm -rf fake_images_ntb

#mv fake_images_tb/results .
#mv results user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.pix2pix_v1.r3.tb.samples
#rm -rf fake_images_tb
