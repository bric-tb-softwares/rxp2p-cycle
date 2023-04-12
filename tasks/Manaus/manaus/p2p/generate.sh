

export MODELS='/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/models/user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.pix2pix_v1.r2/'
python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images --dataset_mode santacasaskfoldextendido2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_train_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images --dataset_mode santacasaskfoldextendido2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_val_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images
python ./generate_synth.py --checkpoints_dir $MODELS  --results_dir fake_images --dataset_mode santacasaskfoldextendido2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/AB/ --model test --n_folds 10 --preprocess resize_and_scale_width  --direction AtoB --netG unet_256 --norm batch  --gpu_ids -1 --gen_test_dataset --n_folds 10 --name genkl --dataset_download_dir $PWD/images

mv fake_images/results .
mv results user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.pix2pix_v1.r2.samples
rm -rf fake_images

