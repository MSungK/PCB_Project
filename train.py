from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.yaml').load('yolov8m.pt')  # build from YAML and transfer weights

# Use the model
# model.train(data="datasets/pcb_roboflow/data.yaml", batch = 16, epochs=100, imgsz=640, device=1, save = True, optimizer='Adam', lr0 = 0.001)  # train the model
model.train(data="data.yaml", batch = 16, epochs=100, imgsz=640, device=1, save = True, optimizer='Adam', lr0 = 0.001)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format