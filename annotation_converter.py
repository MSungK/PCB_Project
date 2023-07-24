import os
import os.path as osp


def converter(norm, x1, y1, x2, y2):
    x_center = ((x2 + x1) / 2) / norm
    y_center = ((y1 + y2) / 2) / norm
    width = (x2 - x1) / norm
    height = (y2 - y1) / norm
    return x_center, y_center, width, height


if __name__ == '__main__': 
    path = 'dataset/DeepPCB/PCBData/'
    label_path = 'dataset/DeepPCB/labels/'
    image_path = 'dataset/DeepPCB/images/'
    dirs = os.listdir(path)
    cnt = 0

    for dir in dirs:

        if osp.splitext(dir)[1] != '':
            continue
        dir_path = osp.join(path, dir)
        sub_dirs = os.listdir(dir_path)

        for sub_dir in sub_dirs:
            if sub_dir.find('not') == -1:
                continue
            txt_path = osp.join(dir_path, sub_dir)
            files = os.listdir(txt_path)
            
            for file in files:
                file_path = osp.join(txt_path, file)
                
                with open(file_path, 'r') as t:
                    lines = t.readlines()

                f = open(osp.join(label_path, file), 'w')

                for line in lines:
                    x1, y1, x2, y2, cls = map(int, line.split(' '))
                    x_center, y_center, width, height = converter(640, x1, y1, x2, y2)
                    label = f'{cls} {x_center} {y_center} {width} {height}\n'
                    f.write(label)
                f.close()
        
                    


