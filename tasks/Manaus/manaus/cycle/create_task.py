import os
basepath = os.getcwd()


taskname = 'user.otto.tavares.Manaus.manaus.cycle_v1.r2'
image = '/home/joao.pinto/public/images/rxp2p-cycle_latest.sif'
repo = "/home/joao.pinto/git_repos/brics/rxp2p-cycle/"


#exec_cmd = f"cd {repo} \n"
#exec_cmd+= f"source setup.sh \n"
exec_cmd = "export PYTHONPATH=/home/joao.pinto/git_repos/brics/rxp2p-cycle \n"
exec_cmd+= "echo 'AKI JOAO'\n"
exec_cmd+= "echo $PYTHONPATH \n"
exec_cmd+= f"cd %JOB_WORKAREA \n"
exec_cmd+= """python /home/joao.pinto/git_repos/brics/rxp2p-cycle/train.py \
--dataroot /home/brics/public/brics_data/Manaus/manaus/raw \
--custom_images_path /home/brics/public/brics_data/Manaus/manaus/raw/images \
--dataset_download_dir /home/joao.pinto/git_repos/brics/rxp2p-cycle/tasks/Manaus/manaus/cycle \
--download_imgs \
--model cycle_gan \
--dataset_mode unalignedskfoldmanaus \
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
--isTB \
--job %IN \
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


