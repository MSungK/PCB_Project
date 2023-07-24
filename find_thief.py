import os 
import os.path as osp


if __name__ == '__main__':
    path = 'datasets/DeepPCB/test/images'
    files = os.listdir(path)
    count = dict()
    complete = dict()
    for file in files:
        file_name = osp.splitext(file)[0]
        file_name = file_name[:file_name.find('_')]
        
        if file_name in count:
            complete[file_name] = True
        else:
            count[file_name] = True
    
    print(len(count.keys()))
    print(len(complete.keys()))

    for key in count.keys():
        if key in complete:
            continue
        else:
            print(key)

                

