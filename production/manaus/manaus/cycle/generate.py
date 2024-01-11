
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
#from util.visualizer import Visualizer
from util import html
from util import stats
import pandas as pd
import numpy as np
import pickle


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    num_synth = 1


    

    opt.name_aux = str.replace(opt.name, 'genkl', '')
    n_sorts = opt.n_folds - 1
    for fold in range(opt.n_folds):
        opt.test = fold
        for val in range(n_sorts):
            opt.sort = val
            opt.name = f'job.test_{fold}.sort_{val}/checkpoints/test_{fold}_sort_{val}/'
            out = f'results/job.test_{fold}.sort_{val}/'
            if opt.gen_train_dataset:
                name_gen = out + 'TRAIN/'
            if opt.gen_val_dataset:
                name_gen = out + 'VAL/'
            if opt.gen_test_dataset:
                name_gen = out +  'TEST/'

            dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
            opt.num_test = len(dataset) * num_synth
            #visualizer = Visualizer(opt)
            model = create_model(opt)      # create a model given opt.model and other options
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            # initialize logger
            #if opt.use_wandb:
            #    wandb_run = wandb.init(project='CycleGAN-and-pix2pix-test', name=opt.name, config=opt) if not wandb.run else wandb.run
            #    wandb_run._label(repo='CycleGAN-and-pix2pix-test')

            # create a website
            web_dir = os.path.join(opt.results_dir, name_gen)#, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
            if opt.load_iter > 0:  # load_iter is 0 by default
                web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
            print('creating web directory', web_dir)
            webpage = html.HTML_GEN(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (name_gen, opt.phase, opt.epoch))
            
            # test with eval mode. This only affects layers like batchnorm and dropout.
            # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
            # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
            if opt.eval:
                model.eval()

            if (num_synth - 1) >= 1:
                for i_synth in range(num_synth):
                    for i, data in enumerate(dataset):
                        #if i >= opt.num_test:  # only apply our model to opt.num_test images.
                        if i + i_synth * len(dataset) >= opt.num_test:
                            break
                        model.set_input(data)  # unpack data from data loader
                        model.test()           # run inference
                        visuals = model.get_current_visuals()  # get image results
                        img_path = model.get_image_paths()     # get image paths
                        img_path = [im[:-4] + '_%s.png'%(i_synth) for im in img_path]
                        print(img_path)
                        if i % 5 == 0:  # save images to an HTML file
                            print('processing (%04d)-th image... %s' % (i, img_path))
                        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
                webpage.save()  # save the HTML
            else:
                for i, data in enumerate(dataset):
                    if i >= opt.num_test:  # only apply our model to opt.num_test images.
                        break
                    model.set_input(data)  # unpack data from data loader
                    model.test()           # run inference
                    visuals = model.get_current_visuals()  # get image results
                    img_path = model.get_image_paths()     # get image paths
                    img_path = [im[:-4] + '_.png' for im in img_path]
                    print(img_path)

                    if i % 5 == 0:  # save images to an HTML file
                        print('processing (%04d)-th image... %s' % (i, img_path))
                    save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
                webpage.save()  # save the HTML
              

