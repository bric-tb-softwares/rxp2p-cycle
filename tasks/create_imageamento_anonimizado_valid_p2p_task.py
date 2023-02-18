import os
basepath = os.getcwd()


taskname = 'user.otto.tavares.SantaCasa_imageamento_anonimizado_valid.pix2pix_v1.r1'
image = '/home/joao.pinto/public/images/rxp2p-cycle_latest.sif'


exec_cmd = """cd /app && export PYTHONPATH=$PWD:$PYTHONPATH && cd %JOB_WORKAREA \n
python /app/train.py \
--dataroot /home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/AB/ \
--dataset_download_dir /home/otto.tavares/public/iltbi/train/images \
--model pix2pix \
--dataset_mode santacasaskfoldextendido \
--token {TOKEN} \
--name check_%JOB_NAME \
--preprocess resize_and_scale_width \
--use_wandb \
--wandb_fold_id check_%JOB_NAME \
--wandb_entity brics-tb  \
--gpu_ids %CUDA_VISIBLE_DEVICES \
--n_epochs 500 \
--n_epochs_decay 500 \
--load_size 256 \
--crop_size 256 \
--job %IN \
--project %JOB_TASKNAME \
"""




exec_cmd = exec_cmd.format(TOKEN='b16fe0fc92088c4840a98160f3848839e68b1148')


command = """maestro.py task create \
  -t {TASK} \
  -i {PATH}/jobs \
  --exec "{EXEC}" \
  --image {IMAGE} \
  --dry_run \
  """


command=command.format(PATH=basepath,EXEC=exec_cmd, TASK=taskname, IMAGE=image)


os.system(command)


