from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8m.yaml').load('yolov8m.pt')  # build from YAML and transfer weights

    # Use the model
    model.train(data="/workspace/minsung/yolov8/datasets/roboflow_9865/data.yaml", batch = 128, epochs=100, imgsz=640, device=[0, 1], save = True, optimizer='Adam', lr0=1e-2, 
                patience=10, save_period=1, workers = 8, )  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    path = model.export(format="onnx")  # export the model to ONNX format