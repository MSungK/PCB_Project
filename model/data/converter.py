import os
import os.path as osp

# normal : 0, delamination : 1, pop-corn : 2, scratch : 3

if __name__ == '__main__':
    root = 'PCB_DEFECTS_230728/GOODPCB'
    files = os.listdir(root)
    f = open('train.txt', 'a')
    for file in files:
        cur_path = osp.join(root, file)
        f.write(f'{cur_path} 0 \n')
    f.close()


