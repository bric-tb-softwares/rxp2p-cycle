"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from util.client_dataset import dorothy_dataset
from PIL import Image, ImageOps
from util.stratified_kfold import stratified_train_val_test_splits_bins
from util.util import prepare_my_table
import pandas as pd
import os, sys
import random
import pickle
import requests
import json
import traceback


class UnalignedSkfoldCmanausDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        
        #clinical_path = opt.dataroot + '/ClinicalReadings'
        if opt.isTrain is False:
            opt.train_dataset = False
        clinical_path = opt.dataroot + '/raw/clinical'
        if opt.custom_masks_path is None:
            masks_path = opt.dataroot + '/A'
        else:
            masks_path = opt.custom_masks_path
        if opt.custom_images_path is None:
            images_path = opt.dataroot + '/B'
        else:
            images_path = opt.custom_images_path
        if opt.custom_paired_path is None:
            paired_path = opt.dataroot + '/AB'
        else:
            paired_path = opt.custom_paired_path

        if opt.generate_paths_data_csv:
            df = prepare_my_table(clinical_path, images_path, masks_path, paired_path, combine=opt.dataset_action)
            #add opt save file path csv for database information
            df.to_csv('...')
        else:
            try:
                df_iltbi = dorothy_dataset(opt.token, 'imageamento_anonimizado_valid', opt.download_imgs, opt.dataset_download_dir)
                df_iltbi = df_iltbi.sort_values('project_id')
                df_manaus = dorothy_dataset(opt.token, 'c_manaus', opt.download_imgs, opt.dataset_download_dir)
                df_manaus = df_manaus.sort_values('project_id')
                df_manaus = df_manaus.reset_index()
                print(df_manaus)
            except Exception as ex:
                print(ex)
                print('Failed to open dorothy database')
                print('importing metada saved from last trial')
                df_manaus = pd.read_csv('/home/brics/public/brics_data/Manaus/c_manaus/raw/Manaus_c_manaus_table_from_raw.csv')
                df_manaus.drop("Unnamed: 0", axis=1, inplace=True)
                df_iltbi = pd.read_csv('/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/SantaCasa_imageamento_anonimizado_valid_table_from_raw.csv')
                df_iltbi.drop("Unnamed: 0", axis=1, inplace=True)
                df_iltbi = df_iltbi.sort_values('project_id')
                #traceback.print_exc()
        ##shenzhen dataset splitting in partitions
        #splits = stratified_train_val_test_splits_bins(df_manaus, opt.n_folds, opt.seed)[opt.test]
        splits = pickle.load(open('/home/brics/public/brics_data/Manaus/c_manaus/raw/splits.pkl','rb'))[opt.test]
        ##importing partition defined as seed for the project
        
        training_data = df_manaus.iloc[splits[opt.sort][0]]
        validation_data = df_manaus.iloc[splits[opt.sort][1]]
        
        path_santa_casa_anonimizado_valid = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/splits.pkl'
        particao_santa_casa_anonimizado_valid = open(path_santa_casa_anonimizado_valid, "rb")
        particao_iltbi = pickle.load(particao_santa_casa_anonimizado_valid)
        particao_santa_casa_anonimizado_valid.close()                
        splits_iltbi = particao_iltbi[opt.test]

        self.training_data_iltbi = df_iltbi.iloc[splits_iltbi[opt.sort][0]]
        self.validation_data_iltbi = df_iltbi.iloc[splits_iltbi[opt.sort][1]]
        
        if (opt.isTB == True):
            self.train_tb = training_data.loc[df_manaus.target == 1]
            self.val_tb = validation_data.loc[df_manaus.target == 1]
            if opt.train_dataset:
                self.B_paths = [img_path for img_path in self.train_tb['image_path'].tolist() if img_path != '']
            else:
                self.B_paths = [img_path for img_path in self.val_tb['image_path'].tolist() if img_path != '']
        else:
            self.train_ntb = training_data.loc[df_manaus.target == 0]
            self.val_ntb = validation_data.loc[df_manaus.target == 0]
            if opt.train_dataset:
                self.B_paths = [img_path for img_path in self.train_ntb['image_path'].tolist() if img_path != '']
            else:
                self.B_paths = [img_path for img_path in self.val_ntb['image_path'].tolist() if img_path != '']
        
        #iltbi dataset
        if opt.train_dataset:
            self.A_paths = [img_path for img_path in self.training_data_iltbi['image_path'].tolist() if img_path != '']
        else:
            self.A_paths = [img_path for img_path in self.validation_data_iltbi['image_path'].tolist() if img_path != '']
        
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc     # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        A_img = ImageOps.exif_transpose(A_img)
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
