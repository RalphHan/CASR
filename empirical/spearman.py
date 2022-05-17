import cv2
import numpy as np
import os
import datasets
import random
import torch
from tqdm import tqdm
import scipy.stats
from collections import defaultdict

def compute_difficulty(src):
    state=src.reshape(9,9)
    diffs=[]
    for i,x in enumerate(src):
        if x==0:
            row,col=i//9,i%9
            row_d=(state[row]==0).sum()
            col_d=(state[col]==0).sum()
            hou_d=(state[row//3*3:row//3*3+3,col//3*3:col//3*3+3]==0).sum()
            diff=(row_d-1)*(col_d-1)*(hou_d-1)
            diffs.append(diff)
    return diffs

if __name__=='__main__':
    path='output/bart_base_sudoku_bs64'
    dataset=datasets.load_dataset(path='csv',
                          data_files={
                              k: os.path.join('data/sudoku',f'sudoku_{k}.csv') for k in ['test']})["test"]
    difficulty=[]
    order=[]
    for casstep in range(5):
        pred=torch.load(os.path.join(path,f'cas_{casstep}/cas_test_generation.pk'))
        cc=0
        for gt,pd in zip(tqdm(dataset),pred):
            src = np.int64(list(gt['quizzes']))
            tgt = np.int64(list(gt['solutions']))
            tgt[src==0]+=9
            if casstep==0:
                difficulty.extend(compute_difficulty(src))
            pd=pd-3
            od=(pd == tgt)[src==0]
            if casstep==0:
                order.extend((5-od*5).tolist())
            else:
                for i in range(len(od)):
                    if od[i]:
                        order[cc + i]=min(casstep,order[cc + i])
                    else:
                        order[cc+i]=5
                cc+=len(od)
        if casstep == 0:
            order=np.int64(order)
            difficulty=np.int64(difficulty)
    print('spearman:',scipy.stats.spearmanr(order, difficulty).correlation)
    gather=defaultdict(list)
    for k,v in zip(order,difficulty):
        gather[k].append(v)

    for k in range(6):
        print(k,np.mean(gather[k]),len(gather[k])/len(difficulty))
    np.random.seed(12321)
    base_sp=[]
    for i in tqdm(range(100)):
        np.random.shuffle(order)
        base_sp.append(scipy.stats.spearmanr(order, difficulty).correlation)
    print('base:',np.mean(base_sp),np.std(base_sp))

# spearman: 0.0736856769363596
# 0 82.80283392828014 0.635360611354119
# 1 86.95408815633604 0.18482097553676394
# 2 88.5986455092925 0.08502936088518841
# 3 89.83256840730941 0.04253184807199044
# 4 90.86453047831675 0.025062195566004024
# 5 92.21165761314306 0.02719500858593416
#
# base: -1.4335261343236081e-05 0.0005399796503994309