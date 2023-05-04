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
from PIL import Image
from util.stratified_kfold import stratified_train_val_test_splits_bins
from util.util import prepare_my_table
from util.client_dataset import dorothy_dataset
import pandas as pd
import os, sys
import pickle
import requests
import json


class ManausSkfold2GeneratorDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    #@staticmethod
    #def modify_commandline_options(parser, is_train):
    #    """Add new dataset-specific options, and rewrite default values for existing options.

    #    Parameters:
    #        parser          -- original option parser
    #        is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    #    Returns:
    #        the modified parser.
    #    """
    #    #parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
    #    #parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        #return parser

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
        
        if opt.isTrain is False and (opt.gen_val_dataset is True or opt.gen_test_dataset is True):
            opt.train_dataset = False
            if opt.gen_val_dataset is True:
                self.gen_val = True
                self.gen_test = False
                
            if opt.gen_test_dataset is True:
                self.gen_val = False
                self.gen_test = True
              
        if opt.isTrain is False and opt.gen_val_dataset is False and opt.gen_test_dataset is False and opt.gen_train_dataset is False:
            opt.train_dataset = False
        if opt.isTrain is False and opt.gen_train_dataset is True:
            opt.train_dataset = True

        try:
            df_manaus = dorothy_dataset(opt.token, 'manaus', opt.download_imgs, opt.dataset_download_dir)
            df_manaus = df_manaus.sort_values('project_id')
            #df_manaus.to_csv(opt.checkpoints_dir + '/imgs_metadata.csv')
        except Exception as ex:
            print(ex)
            print('importing metada saved from last trial')
            #df_manaus = pd.read_csv(opt.checkpoints_dir + 'imgs_metadata.csv')
            #df_manaus = df_manaus.sort_values('project_id')

        
        path_manaus = '/home/brics/public/brics_data/Manaus/manaus/raw/splits.pkl'
        masks_available = pd.read_csv('/home/brics/public/brics_data/Manaus/manaus/masks/dicionario_masks_manaus.csv', sep = ";")[['project_id','binary_limiar']].dropna()
        masks_available = masks_available.sort_values('project_id')

        particao_manaus= open(path_manaus, "rb")
        particao_iltbi = pickle.load(particao_manaus)
        particao_manaus.close()
        splits = particao_iltbi[opt.test]
        
        training_data = df_manaus.iloc[splits[opt.sort][0]]
        validation_data = df_manaus.iloc[splits[opt.sort][1]]
        test_data = df_manaus.iloc[splits[opt.sort][2]]

        if (opt.isTB == True):
            self.train_tb = training_data.loc[df_manaus.target == 1]
            self.val_tb = validation_data.loc[df_manaus.target == 1]
            self.test_tb = test_data.loc[df_manaus.target == 1]
            if opt.train_dataset:
                print('Generating Manaus TB-TRAIN data: ' + str(len(self.train_tb['project_id'].tolist())) + ' imgs')
                self.AB_paths = [opt.dataroot + img_nm + '.png' for img_nm in self.train_tb['project_id'].tolist() if img_nm in masks_available['project_id'].str.replace('mask_|.png','').tolist()]
            else:
                if self.gen_val:
                    print('Generating Manaus TB-VAL data: ' + str(len(self.val_tb['project_id'].tolist())) + ' imgs')        
                    self.AB_paths = [opt.dataroot + img_nm + '.png' for img_nm in self.val_tb['project_id'].tolist() if img_nm in masks_available['project_id'].str.replace('mask_|.png','').tolist()]
                if self.gen_test:
                    print('Generating Manaus TB-TEST data: ' + str(len(self.test_tb['project_id'].tolist())) + ' imgs')
                    self.AB_paths = [opt.dataroot + img_nm + '.png' for img_nm in self.test_tb['project_id'].tolist() if img_nm in masks_available['project_id'].str.replace('mask_|.png','').tolist()]
            
        else:
            self.train_ntb = training_data.loc[df_manaus.target == 0]
            self.val_ntb = validation_data.loc[df_manaus.target == 0]
            self.test_ntb = test_data.loc[df_manaus.target == 0]
            if opt.train_dataset:
                print('Generating Manaus NTB-TRAIN data: ' + str(len(self.train_ntb['project_id'].tolist())) + ' imgs')
                self.AB_paths = [opt.dataroot + img_nm + '.png' for img_nm in self.train_ntb['project_id'].tolist() if img_nm in masks_available['project_id'].str.replace('mask_|.png','').tolist()]
            else:
                if self.gen_val:
                    print('Generating Manaus NTB-VAL data: ' + str(len(self.val_ntb['project_id'].tolist())) + ' imgs')
                    self.AB_paths = [opt.dataroot + img_nm + '.png' for img_nm in self.val_ntb['project_id'].tolist() if img_nm in masks_available['project_id'].str.replace('mask_|.png','').tolist()]
                if self.gen_test:
                    print('Generating Manaus NTB-TEST data: ' + str(len(self.test_ntb['project_id'].tolist())) + ' imgs')
                    self.AB_paths = [opt.dataroot + img_nm + '.png' for img_nm in self.test_ntb['project_id'].tolist() if img_nm in masks_available['project_id'].str.replace('mask_|.png','').tolist()]
               



        ####
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        #base_data_raw_path = '/Users/ottotavares/Documents/COPPE/projetoTB/China/CXR_png/unaligned'

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
