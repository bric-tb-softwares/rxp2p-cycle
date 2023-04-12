import requests
import json
import pandas as pd
import os

def access_dorothy(token, dataset_name):
    header = { "Authorization": 'Token '+ str(token)}
    response = requests.get('https://dorothy-image.lps.ufrj.br/images/?search={DATASET}'.format(DATASET = dataset_name), 
                    headers=header)
    data = json.loads(response.content)
    return data, header


def download_dorothy(dataset_download_dir, dataset_name, data, metadata, header):
    imageamento_exists = os.path.exists(dataset_download_dir + f"/{dataset_name}/")
    if not imageamento_exists:
        os.makedirs(dataset_download_dir + f"/{dataset_name}/")
        imageamento_exists = os.path.exists(dataset_download_dir + f"/{dataset_name}/")
    if imageamento_exists:
        l_images = os.listdir(dataset_download_dir + f"/{dataset_name}/")
        if len(l_images) == len(metadata['project_id']):
            print(f'download {dataset_name} images from dorothy is not necessary')
        else:
            if len(l_images) == 0:
                print(f'first time downloading dorothy images for {dataset_name}')
            else:
                print(f'refreshing dorothy images for {dataset_name} and a new partiton must be definied')
            print(f'downloading images from dorothy for {dataset_name}')
            for img in data:
                file = open(f"{dataset_download_dir}/{dataset_name}/{img['project_id']}.png","wb")
                response = requests.get(img['image_url'], headers=header)
                file.write(response.content)
                file.close()

def dorothy_dataset(token, dataset_name, is_download_imgs, dataset_download_dir):
    data, header = access_dorothy(token, dataset_name)
    imgs_ = {
            'dataset_name': [],
            'target': [],
            'image_url': [],
            'image_path':[],
            'project_id': [],
            'insertion_date': [],
            'metadata': [],
            'date_acquisition': [],
            'number_reports': [],
            }
    n_imgs = 0
    for img in data:
        image_path = dataset_download_dir+ '/%s'%(dataset_name)+'/%s'%(img['project_id'])+'.png' 
        if img['dataset_name'] == 'imageamento_anonimizado_valid' or img['dataset_name'] == 'imageamento' or img['dataset_name'] == 'complete_imageamento_anonimizado_valid':
            imgs_['target'].append(0)
        else:
            imgs_['target'].append(int(img['metadata']['has_tb']))
        imgs_['dataset_name'].append(img['dataset_name'])
        imgs_['image_url'].append(img['image_url'])
        imgs_['project_id'].append(img['project_id'])
        imgs_['image_path'].append(image_path)
        #imgs_['image_path'].append(img[''])
        imgs_['insertion_date'].append(img['insertion_date'])
        imgs_['metadata'].append(img['metadata'])
        imgs_['date_acquisition'].append(img['date_acquisition'])
        imgs_['number_reports'].append(img['number_reports'])
        n_imgs += 1
    df_data = pd.DataFrame.from_dict(imgs_)
    df_data = df_data.sort_values('project_id')
    if is_download_imgs:
        download_dorothy(dataset_download_dir, dataset_name, data, df_data, header)
    return df_data
    
