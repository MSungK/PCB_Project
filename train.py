from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Load a model

    model = YOLO('yolov8m.pt') # build from YAML and transfer weights
        
    backbone_path = 'model/yes_normal/min_err.pt'
    param = torch.load(backbone_path, map_location=f'cuda:{torch.cuda.current_device()}')
    print(f'load backbone: {backbone_path}')
    from collections import OrderedDict

    new_state_dict = OrderedDict()

    for k, v in param['model'].items():
        if 'backbone' in k:
            new_state_dict[k[9:]] = v

    cnt = 0

    for name, param in model.state_dict().items():
        # name : 'model.model.' ~
        # param.keys() : 'backbone.' ~
        if name[12:] in new_state_dict.keys():
            print(f'before: {torch.sum(torch.ne(param, new_state_dict[name[12:]].cpu())).item()}')
            param = new_state_dict[name[12:]]
            print(f'after: {torch.sum(torch.ne(param, new_state_dict[name[12:]])).item()}')
            cnt += 1
    
    print(cnt)
    print(len(new_state_dict.keys()))

    model.train(data="/workspace/Minsung/PCB_Project/minji2/data.yaml", 
                batch=64, epochs=500, imgsz=640,
                device=[0, 1], save = True, optimizer='AdamW', 
                lr0=1e-4, weight_decay=1e-4, 
                patience=30, save_period=30, workers=4, 
                mixup=0.2)  # train the model

    # model.train(data='/workspace/Minsung/PCB_Project/minji2/data.yaml',
    #             batch=128, epochs=300, imgsz=640, device=[2], save=True,
    #             lr0=1e-4, weight_decay=1e-6,
    #             optimizer='AdamW', patience=50, save_period=10, workers=4)
    
    metrics = model.val()  
    

