import os
basepath = os.getcwd()

taskname = 'user.otto.tavares.Manaus.c_manaus.pix2pix_v1.notb.r1'
image = '/mnt/brics_data/images/p2p-cycle_base.sif'
repo_path = '/home/otto.tavares/public/iltbi/rxp2p-cycle'
jobs_path = '/home/otto.tavares/public/iltbi/rxp2p-cycle/production/manaus/c_manaus/p2p/jobs'
binds = {'/mnt/brics_data/': '/mnt/brics_data/', '/home/brics/':'/home/brics'}
token = 'b16fe0fc92088c4840a98160f3848839e68b1148'


exec_cmd = """
cd {REPO_PATH} && source envs.sh && source activate.sh\n
cd %JOB_WORKAREA\n
python {REPO_PATH}/train.py \
--dataroot /home/brics/public/brics_data/Manaus/c_manaus/AB/ \
--dataset_download_dir /home/otto.tavares/public/iltbi/train/images \
--model pix2pix \
--dataset_mode cmanausskfold \
--token {TOKEN} \
--name %JOB_NAME \
--preprocess resize_and_scale_width \
--use_wandb \
--wandb_fold_id %JOB_NAME \
--wandb_entity otavares  \
--gpu_ids %CUDA_VISIBLE_DEVICES \
--n_epochs 500 \
--n_epochs_decay 500 \
--load_size 256 \
--crop_size 256 \
--save_epoch_freq 200 \
--job %IN \
--project %JOB_TASKNAME
"""

exec_cmd = exec_cmd.format(TOKEN=token, REPO_PATH = repo_path)


command = """maestro task create \
  -t {TASK} \
  -i {JOBS_PATH} \
  --exec "{EXEC}" \
  --image "{IMAGE}" \
  --partition gpu \
  --binds "{BINDS}" \
  """


command=command.format(EXEC=exec_cmd, TASK=taskname, IMAGE=image, JOBS_PATH = jobs_path, BINDS = str(binds))


os.system(command)