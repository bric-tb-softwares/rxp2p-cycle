import os

taskname = 'test'


exec_cmd = """train.py \
--dataroot /home/brics/public/brics_data/Shenzhen \
--dataset_download_dir /home/otto.tavares/public/iltbi/train/images \
--download_imgs \
--custom_images_path /home/brics/public/brics_data/Shenzhen/raw/images \
--model cycle_gan \
--dataset_mode unalignedskfoldextendido \
--token ... \
--project cycle_gan_shenzhen_imageamento_extendido \
--name check_test_sort_00 --preprocess resize_and_scale_width \
--use_wandb --wandb_fold_id check_test_sort_00_v0 \
--no_flip  \
--wandb_entity otavares  \
--gpu_ids 0 \
--n_epochs 100 \
--n_epochs_decay 100 \
--load_size 256 \
--crop_size 256 \
--isTB \
--job jobs/job.test_0.sort_0.json \
--test 0 --sort 0 \
--project {TASK} \

"""



#exec_cmd+= "python {PATH}/run.py -j %IN -i {DATA} -t 1 && "
#exec_cmd+= ". {PATH}/end.sh"

#exec_cmd = exec_cmd.format(PATH=basepath, DATA=datapath)


cmd = exec_cmd.format(TASK=taskname)
print(cmd)
os.system(cmd)

#command = """maestro.py task create \
#  -v {PATH} \
#  -t user.jodafons.task.Shenzhen.wgan.v1_tb.test_0.r1 \
#  -i {PATH}/jobs \
#  --exec "{EXEC}" \
#  """
#
#
#
#cmd = command.format(PATH=path,EXEC=exec_cmd)
#print(cmd)
#os.system(cmd)


