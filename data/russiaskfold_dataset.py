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
import pandas as pd
import os, sys


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
        if img['dataset_name'] == 'imageamento':
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
    if opt.download_imgs:
        if dataset_name == 'russia':
            russia_exists = os.path.exists(opt.dataset_download_dir + '/russia/')
            if not russia_exists:
                os.makedirs(opt.dataset_download_dir + '/russia/')
                russia_exists = os.path.exists(opt.dataset_download_dir + '/russia/')
            if russia_exists:
                l_images = os.listdir(opt.dataset_download_dir + '/russia/')
                if len(l_images) == len(df_data['project_id']):
                    print('download russia images from dorothy is not necessary')
                else:
                    if len(l_images) == 0:
                        print('first time downloading dorothy images for russia')
                    else:
                        print('refreshing dorothy images for russia and a new partiton must be definied')
                    print('downloading images from dorothy for russia')
                    for img in data:
                        file = open(f"{opt.dataset_download_dir}/russia/{img['project_id']}.jpg","wb")
                        response = requests.get(img['image_url'], headers=header)
                        file.write(response.content)
                        file.close()
    return df_data

class RussiaSkfoldDataset(BaseDataset):
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
        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        ####

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

        #df = pd.read_csv('/home/otto.tavares/public/iltbi/train/pares_imageamento_santacasa.csv', sep = ';')
        #df.sort_values(by=['images'])
        #df['target'] = 0
        #print(df)
        try:
            df = dorothy_dataset(opt, 'russia')
            df = df.sort_values('project_id')
        except Exception as ex:
            print(ex)
            print('importing metada saved from last trial')
#            df = pd.read_csv('/home/otto.tavares/public/iltbi/particao/Shenzhen_pix2pix_table_from_raw.csv')

        splits = stratified_train_val_test_splits_bins(df, opt.n_folds, opt.seed)[opt.test]
        training_data = df.iloc[splits[opt.sort][0]]
        validation_data = df.iloc[splits[opt.sort][1]]

        
        self.train_tb = training_data.loc[df.all_together == 'seg']
        self.val_tb = validation_data.loc[df.all_together == 'seg']
            #print(type(self.train_tb['paired_image_path']))
        if opt.train_dataset:
            self.AB_paths = [opt.dataroot + img_path for img_path in self.train_tb['images'].tolist()]
        else:
            self.AB_paths = [opt.dataroot + img_path for img_path in self.val_tb['images'].tolist()]

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
