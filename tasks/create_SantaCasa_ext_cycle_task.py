import os
basepath = os.getcwd()


taskname = 'user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.cycle.v1'
image = '/home/joao.pinto/public/images/rxp2p-cycle_latest.sif'

exec_cmd = """nvidia-smi\n source /setup_envs.sh \n
train.py \
--dataroot /home/brics/public/brics_data/Shenzhen \
--dataset_download_dir /home/otto.tavares/public/iltbi/train/images \
--download_imgs \
--custom_images_path /home/brics/public/brics_data/Shenzhen/raw/images \
--model cycle_gan \
--dataset_mode unalignedskfoldextendido \
--token {TOKEN} \
--project cycle_gan_shenzhen_imageamento_extendido \
--name check_%JOB_NAME \
--preprocess resize_and_scale_width \
--use_wandb \
--wandb_fold_id check_%JOB_NAME \
--no_flip  \
--wandb_entity brics-tb  \
--gpu_ids 0 \
--n_epochs 100 \
--n_epochs_decay 100 \
--load_size 256 \
--crop_size 256 \
--isTB \
--job %IN \
--project %JOB_TASKNAME \
"""




cmd = exec_cmd.format(TOKEN='b16fe0fc92088c4840a98160f3848839e68b1148')

command = """maestro.py task create \
  -t {TASK} \
  -i {PATH}/jobs \
  --exec "{EXEC}" \
  --image {IMAGE} \
  --dry_run \
  """


command=command.format(PATH=basepath,EXEC=exec_cmd, TASK=taskname, IMAGE=image)
#print(cmd)


os.system(command)


