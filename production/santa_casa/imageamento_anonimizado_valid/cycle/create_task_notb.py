import os
basepath = os.getcwd()


#taskname = 'user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.cycle_v1.r1'
taskname = 'user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.cycle_notb_v1.r5'
image = '/mnt/brics_data/images/p2p-cycle_base.sif'
repo_path = '/home/otto.tavares/public/iltbi/rxp2p-cycle'
jobs_path = '/home/otto.tavares/public/iltbi/rxp2p-cycle/production/santa_casa/imageamento_anonimizado_valid/cycle/jobs'
binds = {'/mnt/brics_data/': '/mnt/brics_data/', '/home/brics/':'/home/brics'}
token = 'b16fe0fc92088c4840a98160f3848839e68b1148'


exec_cmd = """
cd {REPO_PATH} && source envs.sh && source activate.sh\n
cd %JOB_WORKAREA\n
python {REPO_PATH}/train.py \
--dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid \
--dataset_download_dir /home/otto.tavares/public/iltbi/train/images \
--model cycle_gan \
--dataset_mode unalignedskfoldextendido \
--token {TOKEN} \
--name %JOB_NAME \
--preprocess resize_and_scale_width \
--use_wandb \
--wandb_fold_id %JOB_NAME \
--no_flip  \
--wandb_entity otavares  \
--gpu_ids 0 \
--n_epochs 100  \
--n_epochs_decay 100 \
--load_size 256 \
--crop_size 256 \
--save_epoch_freq 50 \
--job %IN \
--project %JOB_TASKNAME \
"""




exec_cmd = exec_cmd.format(TOKEN=token, REPO_PATH = repo_path)


command = """maestro task create \
  -t {TASK} \
  -i {JOBS_PATH} \
  --exec "{EXEC}" \
  --image "{IMAGE}" \
  --partition gpu-large \
  --binds "{BINDS}" \
  """


command=command.format(EXEC=exec_cmd, TASK=taskname, IMAGE=image, JOBS_PATH = jobs_path, BINDS = str(binds))


os.system(command)


