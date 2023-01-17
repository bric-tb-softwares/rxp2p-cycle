import requests
import json
from data.base_dataset import BaseDataset

class ClientDataset(BaseDataset):

    def __init__(self, opt):
        self.__header = { "Authorization": 'Token '+ opt.token}
        self.dataset_name = dataset_name

    def dataset(self, opt, dataset_name):
        response = requests.get('https://dorothy-image.lps.ufrj.br/images/?search={DATASET}'.format(DATASET= self.dataset_name), 
                                headers=self.__header)
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
            image_path = opt.dataset_download_dir+'/%s'%(img['project_id'])+'.png' 
            #if img['dataset_name'] == 'imageamento':
            imgs_['dataset_name'].append(img['dataset_name'])
            imgs_['target'].append(0)
            imgs_['image_url'].append(img['image_url'])
            imgs_['project_id'].append(image_path)
            imgs_['insertion_date'].append(img['insertion_date'])
            imgs_['metadata'].append(img['metadata'])
            imgs_['date_acquisition'].append(img['date_acquisition'])
            imgs_['number_reports'].append(img['number_reports'])
            n_imgs += 1
        df_data = pd.DataFrame.from_dict(imgs_)
        df_data = df_data.sort_values('project_id')
        df_iltbi = pd.read_csv('/home/otto.tavares/iltbi/particao/imageamento_metadados_iltbi.csv')
        
        if n_imgs != len(df_iltbi['project_id']):
            print('Renovar a partição')
        
        if opt.download_imgs:
            for img in data:
                file = open(f"./{dataset}/{img['project_id']}.jpg","wb")
                response = requests.get(img['image_url'], headers=self.__header)
                file.write(response.content)
                file.close()

        return df_data
        
