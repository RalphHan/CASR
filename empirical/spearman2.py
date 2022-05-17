import cv2
import numpy as np
import os
import datasets
import random
import torch
from tqdm import tqdm
import scipy.stats
from collections import defaultdict


if __name__=='__main__':
    path='output/bart_base_sudoku_bs64'
    dataset=datasets.load_dataset(path='csv',
                          data_files={
                              k: os.path.join('data/sudoku',f'sudoku_{k}.csv') for k in ['test']})["test"]
    difficulty=[]
    order=[]
    accs=[]
    for casstep in range(5):
        pred=torch.load(os.path.join(path,f'cas_{casstep}/cas_test_generation.pk'))
        tps=np.zeros(9,dtype=np.int64)
        cnts=np.zeros(9,dtype=np.int64)
        for gt,pd in zip(tqdm(dataset),pred):
            src = np.int64(list(gt['quizzes']))
            tgt = np.int64(list(gt['solutions']))
            tgt[src==0]+=9
            pd=pd-3
            tp=((pd == tgt)&(src==0)).reshape(9,9).sum(-1)
            cnt=(src==0).reshape(9,9).sum(-1)
            tps+=tp
            cnts+=cnt
        acc=(tps/cnts).astype(np.float32)
        accs.append(acc)
    accs=np.float32(accs)
    torch.save(accs,'output/bart_base_sudoku_bs64/row_accs.pt')
    print(accs)
# spearman: 0.0736856769363596
# 0 82.80283392828014 0.635360611354119
# 1 86.95408815633604 0.18482097553676394
# 2 88.5986455092925 0.08502936088518841
# 3 89.83256840730941 0.04253184807199044
# 4 90.86453047831675 0.025062195566004024
# 5 92.21165761314306 0.02719500858593416
#
# base: -1.4335261343236081e-05 0.0005399796503994309