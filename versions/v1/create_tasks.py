import os
basepath = os.getcwd()

path = basepath

taskname = 'user.otavares.Shenzhen.pix2pix.v1_tb.r1'



#exec_cmd = ". {PATH}/init.sh && "

exec_cmd = """train.py --job jobs/job.test_0.sort_2.json \
--dataroot /home/jodafons/public/brics_data/Shenzhen \
--custom_paired_path /home/jodafons/public/brics_data/Shenzhen/AB \
--custom_masks_path /home/jodafons/public/brics_data/CXR_images_masks \
--custom_images_path /home/jodafons/public/brics_data/Shenzhen/raw/images \
--dataset_mode skfold \
--model pix2pix \
--direction AtoB \
--preprocess resize_and_scale_width \
--load_size 256 \
--gpu_ids -1 \
--n_folds 10 \
--n_epochs 500 \
--n_epochs_decay 500 \
--generate_paths_data_csv True  \
--save_epoch_freq 200 \
--display_id 1 \
--project {TASK} \
--wandb_fold_id test_0_sort_2 \
--wandb_entity brics-tb \
--name test_0_sort_2 \
--isTB \
--checkpoints_dir $PWD"""
#--use_wandb \


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


