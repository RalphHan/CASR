import sys
with open(sys.argv[1]) as f:
    lines=f.readlines()
with open(sys.argv[1],'w') as f:
    for line in lines:
        line=line.strip().split()
        line=line[:int(sys.argv[2])]
        line=' '.join(line)
        f.write(line+'\n')
