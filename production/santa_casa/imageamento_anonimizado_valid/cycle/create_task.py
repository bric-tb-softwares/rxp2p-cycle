import os
basepath = os.getcwd()


#taskname = 'user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.cycle_v1.r1'
taskname = 'user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.cycle_v1.r1_exemplo'
image = '/mnt/brics_data/images/p2p-cycle_base.sif'




exec_cmd = """cd /app && export PYTHONPATH=$PWD:$PYTHONPATH && cd %JOB_WORKAREA \n
python /app/train.py \
--dataroot /home/brics/public/brics_data/Shenzhen \
--dataset_download_dir /home/otto.tavares/public/iltbi/train/images \
--download_imgs \
--custom_images_path /home/brics/public/brics_data/Shenzhen/raw/images \
--model cycle_gan \
--dataset_mode unalignedskfoldextendido \
--token {TOKEN} \
--name check_%JOB_NAME \
--preprocess resize_and_scale_width \
--use_wandb \
--wandb_fold_id check_%JOB_NAME \
--no_flip  \
--wandb_entity brics-tb  \
--gpu_ids %CUDA_VISIBLE_DEVICES \
--n_epochs 100 \
--n_epochs_decay 100 \
--load_size 256 \
--crop_size 256 \
--job %IN \
--job isTB \
--project %JOB_TASKNAME \
"""




exec_cmd = exec_cmd.format(TOKEN='b16fe0fc92088c4840a98160f3848839e68b1148')


command = """maestro.py task create \
  -t {TASK} \
  -i {PATH}/jobs \
  --exec "{EXEC}" \
  --image {IMAGE} \
  --skip_test
  """


command=command.format(PATH=basepath,EXEC=exec_cmd, TASK=taskname, IMAGE=image)


os.system(command)


