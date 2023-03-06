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


class CycleSantaCasaSkfold2GeneratorDataset(BaseDataset):

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
        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        ####
        self.__header = { "Authorization": 'Token '+ opt.token}

        #clinical_path = opt.dataroot + '/ClinicalReadings'
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
            df.to_csv('...')
        else:
            #df = pd.read_csv('/home/otto.tavares/Shenzhen_pix2pix_table_from_raw.csv')
            #df.drop("Unnamed: 0", axis=1, inplace=True)
            #df_iltbi = pd.read_csv('/home/otto.tavares/iltbi/particao/imageamento_metadados_iltbi.csv')
            try:
                df = dorothy_dataset(opt, 'china')
                df = df.sort_values('project_id')
                df_iltbi = dorothy_dataset(opt, 'imageamento')
                df_iltbi = df_iltbi.sort_values('project_id')
            except Exception as ex:
                print(ex)
                print('Dorothy out of service or unsolved bug')
        
        #shenzhen dataset splitting in partitions
        #splits = stratified_train_val_test_splits_bins(df, opt.n_folds, opt.seed)[opt.test]
        #importing partition defined as seed for the project
        path_shenzhen = '/home/otto.tavares/public/particao.pkl'
        particao_shenzhen_file = open(path_shenzhen, "rb")
        particao_shenzhen = pickle.load(particao_shenzhen_file)
        particao_shenzhen_file.close()
        splits = particao_shenzhen[opt.test]
        training_data = df.iloc[splits[opt.sort][0]]
        validation_data = df.iloc[splits[opt.sort][1]]
        test_data = df.iloc[splits[opt.sort][2]]

        #iltbi dataset splitting in partitions
        path_santa_casa_anonimizado_valid = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/raw/splits.pkl'
        particao_iltbi_file = open(path_iltbi, "rb")
        particao_iltbi = pickle.load(particao_iltbi_file)
        particao_iltbi_file.close()
        splits_iltbi = particao_iltbi[opt.test]
        self.training_data_iltbi = df_iltbi.iloc[splits_iltbi[opt.sort][0]]
        self.validation_data_iltbi = df_iltbi.iloc[splits_iltbi[opt.sort][1]]
        self.test_data_iltbi = df_iltbi.iloc[splits_iltbi[opt.sort][2]]

        if (opt.isTB == True):
            self.train_tb = training_data.loc[df.target == 1]
            self.val_tb = validation_data.loc[df.target == 1]
            self.test_tb = test_data.loc[df.target == 1]
            if opt.train_dataset:
                print('Generating TB and TRAIN data: ' + str(len(self.train_tb['project_id'].tolist())) + ' imgs')
                self.B_paths = [opt.dataroot + img_path +'.png' for img_path in self.train_tb['project_id'].tolist() if img_path != '']
            else:
                if self.gen_val:
                    print('Generating TB and VAL data: ' + str(len(self.val_tb['project_id'].tolist())) + ' imgs')
                    self.B_paths = [opt.dataroot + img_path +'.png' for img_path in self.val_tb['project_id'].tolist() if img_path != '']
                if self.gen_test:
                    print('Generating TB and TEST data: ' + str(len(self.test_tb['project_id'].tolist())) + ' imgs')
                    self.B_paths = [opt.dataroot + img_path +'.png' for img_path in self.test_tb['project_id'].tolist() if img_path != '']         
        else:
            self.train_ntb = training_data.loc[df.target == 0]
            self.val_ntb = validation_data.loc[df.target == 0]
            self.test_ntb = test_data.loc[df.target == 0]
            if opt.train_dataset:
                self.B_paths = [opt.dataroot + img_path +'.png' for img_path in self.train_ntb['project_id'].tolist() if img_path != '']
            else:
                if self.gen_val:
                    print('Generating TB and VAL data: ' + str(len(self.val_ntb['project_id'].tolist())) + ' imgs')
                    self.B_paths = [opt.dataroot + img_path +'.png' for img_path in self.val_ntb['project_id'].tolist() if img_path != '']
                if self.gen_test:
                    print('Generating TB and TEST data: ' + str(len(self.test_ntb['project_id'].tolist())) + ' imgs')
                    self.B_paths = [opt.dataroot + img_path +'.png' for img_path in self.test_ntb['project_id'].tolist() if img_path != '']         
        
        #iltbi dataset
        if opt.train_dataset:
            self.A_paths = [opt.dataroot + img_path +'.png' for img_path in self.training_data_iltbi['project_id'].tolist() if img_path != '']
        else:
            if self.gen_val:
                self.A_paths = [opt.dataroot + img_path +'.png' for img_path in self.validation_data_iltbi['project_id'].tolist() if img_path != '']
            if self.gen_test:
                self.A_paths = [opt.dataroot + img_path +'.png' for img_path in self.test_data_iltbi['project_id'].tolist() if img_path != '']
        
        
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
