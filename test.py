from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train5/weights/best.pt')  # build from YAML and transfer weights

# Use the model
# model.train(data="datasets/pcb_roboflow/data.yaml", batch = 16, epochs=100, imgsz=640, device=1, save = True, optimizer='Adam', lr0 = 0.001)  # train the model

metrics = model.val(data="data.yaml", batch=64)  # evaluate model performance on the validation set
print(metrics.box.map)
print(metrics.box.map50)
print(metrics.box.map75)
