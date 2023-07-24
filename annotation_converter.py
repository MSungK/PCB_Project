import os
import os.path as osp


def converter(norm, x1, y1, x2, y2):
    x_center = ((x2 + x1) / 2) / norm
    y_center = ((y1 + y2) / 2) / norm
    width = (x2 - x1) / norm
    height = (y2 - y1) / norm
    return x_center, y_center, width, height


def make_dict(trainset_info):
    is_train = dict()
    standard = 'datasets/DeepPCB/PCBData/'

    with open(trainset_info, 'r') as reader:
        lines = reader.readlines()

    for line in lines:
        file1, file2 = line.split(' ')
        is_train[osp.splitext(osp.join(standard,file1))[0]] = True
        is_train[osp.splitext(osp.join(standard,file2))[0]] = True

    return is_train


if __name__ == '__main__': 
    path = 'datasets/DeepPCB/PCBData/'

    trainset_info = 'datasets/DeepPCB/PCBData/trainval.txt'

    train_path = 'datasets/DeepPCB/train/labels'
    test_path = 'datasets/DeepPCB/test/labels'

    train_img_path = 'datasets/DeepPCB/train/images'
    test_img_path = 'datasets/DeepPCB/test/images'

    is_train = make_dict(trainset_info)

    dirs = os.listdir(path)
    cnt = 0

    train_img = 0
    train_label = 0
    test_img = 0
    test_label = 0

    for dir in dirs:

        if osp.splitext(dir)[1] != '':
            continue
        dir_path = osp.join(path, dir)
        sub_dirs = os.listdir(dir_path)

        for sub_dir in sub_dirs:
            if sub_dir.find('not') != -1:
                txt_path = osp.join(dir_path, sub_dir)
                files = os.listdir(txt_path)
                
                for file in files:
                    file_path = osp.join(txt_path, file)
                    check_path = osp.splitext(file_path)[0]
                    with open(file_path, 'r') as t:
                        lines = t.readlines()

                    if check_path in is_train:
                        out = train_path
                        train_label += 1
                    else:
                        out = test_path
                        test_label += 1
                    
                    file_name = osp.splitext(file)[0]

                    test_name = file_name + '_test.txt'
                    temp_name = file_name + '_temp.txt'

                    f = open(osp.join(out, test_name), 'w')

                    for line in lines:
                        x1, y1, x2, y2, cls = map(int, line.split(' '))
                        x_center, y_center, width, height = converter(640, x1, y1, x2, y2)
                        label = f'{cls} {x_center} {y_center} {width} {height}\n'
                        f.write(label)
                    f.close()

                    f = open(osp.join(out, temp_name), 'w')
                    f.write('\n')
                    f.close()

            else:
                img_path = osp.join(dir_path, sub_dir)
                files = os.listdir(img_path)

                for file in files:
                    file_path = osp.join(img_path, file)
                    check_path = file_path[:file_path.find('_')]

                    if check_path in is_train:
                        out = train_img_path
                        train_img += 1
                    else:
                        out = test_img_path
                        test_img += 1
                    
                    out_path = osp.join(out , file)
                    os.system(f'cp {file_path} {out_path}')

    print(f'train_img: {train_img}')
    print(f'train_label: {train_label}')
    print(f'test_img: {test_img}')
    print(f'test_label: {test_label}')


            
                        


