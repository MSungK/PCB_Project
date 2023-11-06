from ultralytics import YOLO
import torch


# Load a model
model = YOLO('best.pt') # build from YAML and transfer weights
model_children_list = list(model.model.children())
backbone = model_children_list[0][:10]
torch.save(backbone.state_dict(), 'backbone.pt')

model = YOLO('yolov8m.pt')
model.load_state_dict(torch.load('backbone.pt'), strict=False)
print(list(model.model.children())[0][:10])


# Use the model
# model.train(data="datasets/pcb_roboflow/data.yaml", batch = 16, epochs=100, imgsz=640, device=1, save = True, optimizer='Adam', lr0 = 0.001)  # train the model

# metrics = model.val(data="data.yaml", batch=64)  # evaluate model performance on the validation set
# print(metrics.box.map)
# print(metrics.box.map50)
# print(metrics.box.map75)
