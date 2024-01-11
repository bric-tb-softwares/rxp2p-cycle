import glob
import pandas as pd
 
dataframe = {
      'dataset_name': [],
      'dataset_type': [],
      'project_id': [],
      'image_path':[],
      'metadata': [],
      'test':[],
      'sort':[],
      'type':[],
      }

basepath = '/home/brics/public/brics_data/Manaus/manaus/fake_images'

#dataset_name = 'user.otto.tavares.Manaus.manaus.pix2pix_v1.notb.r3.samples'
dataset_name = 'user.otto.tavares.Manaus.manaus.pix2pix_v1.tb.r3.samples'
#has_tb = False
has_tb = True


for test in range(10):
    for sort in range(9):

        paths = {
            'train' : dataset_name + f'/job.test_{test}.sort_{sort}' + '/TRAIN',
            'val'   : dataset_name + f'/job.test_{test}.sort_{sort}' + '/VAL',
            'test'  : dataset_name + f'/job.test_{test}.sort_{sort}' + '/TEST',

        }

        project_id = 0
        for key, path in paths.items():

            for f in glob.glob(path + '/*.png'):
                if f[-10:].replace('.png', '') == 'fake_B':


                    abspath = basepath + '/' + path + '/' + f.split('/')[-1]
                    print(abspath)

                    dataframe['image_path'].append(abspath)
                    dataframe['metadata'].append( {'has_tb':has_tb} )
                    dataframe['project_id'].append(project_id)
                    dataframe['test'].append(test)
                    dataframe['sort'].append(sort)
                    dataframe['type'].append(key)
                    dataframe['dataset_name'].append(dataset_name)
                    dataframe['dataset_type'].append('synthetic')






df_data = pd.DataFrame.from_dict(dataframe)
df_data.to_csv(dataset_name+'.csv')
