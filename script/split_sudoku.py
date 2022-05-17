import random
in_path=r'.'
with open(in_path+r'\sudoku.csv') as f:
    lines=f.readlines()
data=lines[1:]
random.seed(12321)
random.shuffle(data)
with open(in_path+r'\sudoku_train.csv','w') as f:
    f.writelines(lines[:1])
    f.writelines(data[:int(len(data)*0.8)])

with open(in_path+r'\sudoku_eval.csv','w') as f:
    f.writelines(lines[:1])
    f.writelines(data[int(len(data) * 0.8):int(len(data) * 0.9)])

with open(in_path+r'\sudoku_test.csv','w') as f:
    f.writelines(lines[:1])
    f.writelines(data[int(len(data) * 0.9):])

