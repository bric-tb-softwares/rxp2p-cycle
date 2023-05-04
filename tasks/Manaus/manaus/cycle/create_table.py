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


# M 2 S TB a
#dataset_name = 'user.otto.tavares.Manaus.manaus.cycle_v1_tb.r2.Manaus_to_SantaCasa.samples'
#dirname = 'fake_A'
#has_tb = True


# S 2 M NTB b
dataset_name = 'user.otto.tavares.Manaus.manaus.cycle_v1_notb.r2.SantaCasa_to_Manaus.samples'
dirname = 'fake_B'
has_tb = False

for test in range(10):
    for sort in range(9):

        paths = {
            'train' : dataset_name + f'/job.test_{test}.sort_{sort}/' + '/TRAIN',
            'val'   : dataset_name + f'/job.test_{test}.sort_{sort}/' + '/VAL',
            'test'  : dataset_name + f'/job.test_{test}.sort_{sort}/' + '/TEST',

        }

        project_id = 0
        for key, path in paths.items():

            for f in glob.glob(path + '/*.png'):
                if f[-10:].replace('.png', '') == dirname:


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
