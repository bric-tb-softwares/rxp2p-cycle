from rxwgan.datasets import Client
import pandas as pd
from tqdm import tqdm
import os



class DownloadDataset(BaseDataset):

    """Download specified dataset from Dorothy"""


    def __init__(self, opt):
        self.service = Client(token= opt.token)
        

    def download(self, dataset_name, output_images, basepath=opt.dataset_download_dir):

   
        output_images = basepath + '/' + output_images
        # Creating output dir    
        os.makedirs(output_images, exist_ok=True)

        dataset = self.service.dataset(dataset_name)


        # template
        d = {
            'dataset_name': [],
            'project_id': [],
            #'target': [],
            #'image_md5':[],
            #'image_url': [],
            'image_path':[],
            'insertion_date': [],
            'metadata': [],
            #'date_acquisition': [],
            #'number_reports': [],
        }

        # Download each image
        for image in tqdm(dataset.list_images()):
            d['dataset_name'].append(image.dataset_name)
            d['project_id'].append(image.project_id)
            image_path = output_images+'/%s'%(image.project_id)+'.png' 
            if not os.path.exists(image_path):
                image.download(image_path)
            d['image_path'].append(image_path)
            d['metadata'].append(image.metadata)
            d['insertion_date'].append(image.insertion_date)

        df = pd.DataFrame(d)
        return df


if __name__ == "__main__":
    import os
    #token = opt.token
    c = DownloadDataset(opt.token)
    df = c.download(opt.dataset_name, opt.dataset_download_dir)





