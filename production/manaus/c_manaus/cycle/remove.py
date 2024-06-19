

import os
import numpy as np

ids = np.arange(0,200,5)
keep = [50, 100, 150, 200]


for test in range(10):
    for sort in range(9):

        # job.test_0.sort_0/checkpoints/test_0_sort_0
        path = f'job.test_{test}.sort_{sort}/checkpoints/check_job.test_{test}_sort_{sort}'

        
        print(path)

        for idx in ids:
            if idx in keep:
                continue
            else:

                os.system( 'rm %s/%d_net_D_A.pth'%(path,idx) )
                os.system( 'rm %s/%d_net_D_B.pth'%(path,idx) )
                os.system( 'rm %s/%d_net_G_A.pth'%(path,idx) )
                os.system( 'rm %s/%d_net_G_B.pth'%(path,idx) )

                

