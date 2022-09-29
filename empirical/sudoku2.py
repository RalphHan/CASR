import cv2
import numpy as np
import os
import random
import torch
from tqdm import tqdm
def draw(prediction,dependency):
    img=np.full((256,256,3),220,dtype=np.uint8)
    for i,c in enumerate(prediction):
        if c==0 or c>9:
            if i not in dependency:
                cv2.rectangle(img,(10+i%9*26, 10+i//9*26),(36+i%9*26, 36+i//9*26),(255,255,255),-1)
            else:
                cv2.rectangle(img, (10 + i % 9 * 26, 10 + i // 9 * 26), (36 + i % 9 * 26, 36 + i // 9 * 26),
                              (255, 180, 180), -1)
        if c>0:
            txt=str((c-1)%9+1)
            color=[(0,0,0),(0,0,255),(0,120,0)][(c-1)//9]
            cv2.putText(img, txt, (18+i%9*26, 30+i//9*26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    for i in range(10):
        cv2.line(img,(10+26*i,10),(10+26*i,245),(0,0,0),2 if i%3==0 else 1)
    for i in range(10):
        cv2.line(img,(10,10+26*i),(245,10+26*i),(0,0,0),2 if i%3==0 else 1)
    return img

if __name__=='__main__':
    import datasets
    path='output/bart_base_sudoku_bs64'
    os.makedirs(os.path.join(path,'pic4'),exist_ok=True)
    gt=datasets.load_dataset(path='csv',
                          data_files={
                              k: os.path.join('data/sudoku',f'sudoku_{k}.csv') for k in ['test']})["test"][96339]
    src = np.int64(list(gt['quizzes']))
    tgt = np.int64(list(gt['solutions']))
    tgt[src == 0] += 18
    preds=[]
    atts=[]
    for casstep in range(5):
        pred=torch.load(os.path.join(path,f'cas_{casstep}/cas_test_generation.pk'))[96339]
        pd = pred - 3
        pd[(src == 0) & (pd + 9 == tgt)] += 9
        preds.append(pd)
        if casstep==0:
            atts.append(None)
        else:
            atts.append(torch.load(os.path.join(path,f'cas_{casstep}/cas_test_generation.pk.96339.att')))

    for i in range(1,5):
        os.makedirs(os.path.join(path, 'pic4',str(i)), exist_ok=True)
        mask=np.where((preds[i-1]!=preds[i])&(preds[i]==tgt))[0]
        for x in mask:
            row_att=atts[i][x]
            row_att[src != 0]=0
            row_att[x]=0
            dependency=np.argsort(row_att)[-5:]
            img_pd = draw(preds[i-1],dependency)
            cv2.imwrite(os.path.join(path, 'pic4', str(i), '%d-%d.png'%(x//9,x%9)), img_pd)

