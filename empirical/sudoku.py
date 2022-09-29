import cv2
import numpy as np
import os
import random
import torch
from tqdm import tqdm
def draw(prediction):
    img=np.full((256,256,3),220,dtype=np.uint8)
    for i,c in enumerate(prediction):
        if c==0 or c>9:
            cv2.rectangle(img,(10+i%9*26, 10+i//9*26),(36+i%9*26, 36+i//9*26),(255,255,255),-1)
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
    os.makedirs(os.path.join(path,'pic3'),exist_ok=True)
    dataset=datasets.load_dataset(path='csv',
                          data_files={
                              k: os.path.join('data/sudoku',f'sudoku_{k}.csv') for k in ['test']})["test"]
    random.seed(12321)
    slice=random.sample(range(len(dataset)),100)
    for casstep in range(5):
        pred=torch.load(os.path.join(path,f'cas_{casstep}/cas_test_generation.pk'))
        for order in tqdm(slice):
            os.makedirs(os.path.join(path, 'pic3', str(order)),exist_ok=True)
            gt=dataset[order]
            src = np.int64(list(gt['quizzes']))
            tgt = np.int64(list(gt['solutions']))
            tgt[src==0]+=18
            if casstep==0:
                img_gt=draw(tgt)
                img_blank=draw(src)
                cv2.imwrite(os.path.join(path, 'pic3', str(order),'gt.png'),img_gt)
                cv2.imwrite(os.path.join(path, 'pic3', str(order),'blank.png'),img_blank)
            pd=pred[order]-3
            pd[(src==0)&(pd+9==tgt)]+=9
            img_pd = draw(pd)
            cv2.imwrite(os.path.join(path, 'pic3', str(order), f'{casstep}.png'), img_pd)

# img=draw([1,2,3,4,5,6,7,8,0,1,2,3,4,5+9,6+18,7,8,0,1+9,2,3+18,4,5,6,7,8,9]*3)
# cv2.imshow('img',img)
# cv2.waitKey(0)