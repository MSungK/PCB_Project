import os
import os.path as osp
from shutil import move

if __name__ == '__main__':
    root = 'PCB_DEFECTS_230728/POP_CORN'
    files = os.listdir(root)
    cnt = 200
    for file in files:
        if '(' in file:
            path = osp.join(root,file)
            out = osp.join(root, f'{cnt}.png')
            move(path, out)
            cnt += 1