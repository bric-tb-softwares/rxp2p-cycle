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

#TO-DO IMPORT ILTBI IMAGES AND RUN CYCLE FROM SHENZHEN TO ILTBI
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from data import client_dataset
from PIL import Image, ImageOps
from util.stratified_kfold import stratified_train_val_test_splits_bins
from util.util import prepare_my_table
import pandas as pd
import os, sys
import random
import pickle
import requests
import json

def dorothy_dataset(opt, dataset_name):
    
    header = { "Authorization": 'Token '+ str(opt.token)}
    response = requests.get('https://dorothy-image.lps.ufrj.br/images/?search={DATASET}'.format(DATASET = dataset_name), 
                        headers=header)

    data = json.loads(response.content)
    imgs_ = {
            'dataset_name': [],
            'target': [],
            'image_url': [],
            'project_id': [],
            'insertion_date': [],
            'metadata': [],
            'date_acquisition': [],
            'number_reports': [],
            }
    n_imgs = 0
    for img in data:
        image_path = opt.dataset_download_dir+ '/%s'%(dataset_name)+'/%s'%(img['project_id'])+'.jpg' 
        if img['dataset_name'] == 'imageamento_anonimizado_valid':
            imgs_['target'].append(0)
        else:
            imgs_['target'].append(int(img['metadata']['has_tb']))
        imgs_['dataset_name'].append(img['dataset_name'])
        imgs_['image_url'].append(img['image_url'])
        imgs_['project_id'].append(image_path)
        imgs_['insertion_date'].append(img['insertion_date'])
        imgs_['metadata'].append(img['metadata'])
        imgs_['date_acquisition'].append(img['date_acquisition'])
        imgs_['number_reports'].append(img['number_reports'])
        n_imgs += 1
    df_data = pd.DataFrame.from_dict(imgs_)
    df_data = df_data.sort_values('project_id')
    #df_iltbi = pd.read_csv('/home/otto.tavares/iltbi/particao/imageamento_metadados_iltbi.csv')

    #if n_imgs != len(df_iltbi['project_id']):
    #    print('Renovar a partição')
    if opt.download_imgs:
        if dataset_name == 'imageamento_anonimizado_valid':
            imageamento_exists = os.path.exists(opt.dataset_download_dir + '/imageamento_anonimizado_valid/')
            if not imageamento_exists:
                os.makedirs(opt.dataset_download_dir + '/imageamento_anonimizado_valid/')
                imageamento_exists = os.path.exists(opt.dataset_download_dir + '/imageamento_anonimizado_valid/')
            if imageamento_exists:
                l_images = os.listdir(opt.dataset_download_dir + '/imageamento_anonimizado_valid/')
                if len(l_images) == len(df_data['project_id']):
                    print('download imageamento anonimizado images from dorothy is not necessary')
                else:
                    if len(l_images) == 0:
                        print('first time downloading dorothy images for imageamento anonimizado')
                    else:
                        print('refreshing dorothy images for imageamento and a new partiton must be definied')
                    print('downloading images from dorothy for imageamento')
                    for img in data:
                        file = open(f"{opt.dataset_download_dir}/imageamento_anonimizado_valid/{img['project_id']}.jpg","wb")
                        response = requests.get(img['image_url'], headers=header)
                        file.write(response.content)
                        file.close()
        if dataset_name == 'china':
            china_exists = os.path.exists(opt.dataset_download_dir + '/china/')
            if not china_exists:
                os.makedirs(opt.dataset_download_dir + '/china/')
                china_exists = os.path.exists(opt.dataset_download_dir + '/china/')
            if china_exists:
                l_images = os.listdir(opt.dataset_download_dir + '/china/')
                if len(l_images) == len(df_data['project_id']):
                    print('download china images from dorothy is not necessary')
                else:
                    if len(l_images) == 0:
                        print('first time downloading dorothy images for imageamento')
                    else:
                        print('refreshing dorothy images for imageamento and a new partiton must be definied')
                    print('downloading images from dorothy for imageamento')
                    for img in data:
                        print('downloading images from dorothy for china')
                        file = open(f"{opt.dataset_download_dir}/china/{img['project_id']}.jpg","wb")
                        response = requests.get(img['image_url'], headers=header)
                        file.write(response.content)
                        file.close()
    return df_data


class UnalignedSkfoldExtendidoDataset(BaseDataset):

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
            df.to_csv('Shenzhen_pix2pix_table_from_raw.csv')
        else:
            #df = pd.read_csv('/home/otto.tavares/Shenzhen_pix2pix_table_from_raw.csv')
            #df.drop("Unnamed: 0", axis=1, inplace=True)
            #df_iltbi = pd.read_csv('/home/otto.tavares/iltbi/particao/imageamento_metadados_iltbi.csv')
            try:
                df = dorothy_dataset(opt, 'china')
                df = df.sort_values('project_id')
                df_iltbi = dorothy_dataset(opt, 'imageamento_anonimizado_valid')
                df_iltbi = df_iltbi.sort_values('project_id')
            except Exception as ex:
                print(ex)
                print('importing metada saved from last trial')
                #df = pd.read_csv('/home/otto.tavares/public/iltbi/particao/Shenzhen_pix2pix_table_from_raw.csv')
                #df.drop("Unnamed: 0", axis=1, inplace=True)
                #df['project_id'] = [img_path.replace('jodafons', 'brics') for img_path in df['raw_image_path'].tolist() if img_path != '']
                #df = df.sort_values('project_id')
                #df_iltbi = pd.read_csv('/home/otto.tavares/public/iltbi/particao/imageamento_metadados_iltbi_atualizado.csv')
                #df_iltbi['project_id'] = [opt.dataset_download_dir + '/imageamento_atualizado'+'/%s'%(img_path)+'.jpg'  for img_path in df_iltbi['project_id'].tolist() if img_path != '']
                #df_iltbi = df_iltbi.sort_values('project_id')

            
            
        
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

        #iltbi dataset splitting in partitions
        #path_iltbi = '/home/otto.tavares/public/iltbi/particao/particao_imageamento_atualizado.pkl'
        #particao_iltbi_file = open(path_iltbi, "rb")
        #particao_iltbi = pickle.load(particao_iltbi_file)
        #particao_iltbi_file.close()
        #splits_iltbi = particao_iltbi[opt.test]
        splits_iltbi = stratified_train_val_test_splits_bins(df_iltbi, opt.n_folds, opt.seed)[opt.test]
        self.training_data_iltbi = df_iltbi.iloc[splits_iltbi[opt.sort][0]]
        self.validation_data_iltbi = df_iltbi.iloc[splits_iltbi[opt.sort][1]]
        
        if (opt.isTB == True):
            self.train_tb = training_data.loc[df.target == 1]
            self.val_tb = validation_data.loc[df.target == 1]
            if opt.train_dataset:
                self.B_paths = [img_path for img_path in self.train_tb['project_id'].tolist() if img_path != '']
            else:
                self.B_paths = [img_path for img_path in self.val_tb['project_id'].tolist() if img_path != '']
        else:
            self.train_ntb = training_data.loc[df.target == 0]
            self.val_ntb = validation_data.loc[df.target == 0]
            if opt.train_dataset:
                self.B_paths = [img_path for img_path in self.train_ntb['project_id'].tolist() if img_path != '']
            else:
                self.B_paths = [img_path for img_path in self.val_ntb['project_id'].tolist() if img_path != '']
        
        #iltbi dataset
        if opt.train_dataset:
            self.A_paths = [img_path for img_path in self.training_data_iltbi['project_id'].tolist() if img_path != '']
        else:
            self.A_paths = [img_path for img_path in self.validation_data_iltbi['project_id'].tolist() if img_path != '']
        
        
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
