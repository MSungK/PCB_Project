from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    img = 'datasets/pcb_roboflow/valid/images/06_short_01_jpg.rf.f6978b172d9889903e996f49a49010df.jpg'
    model = YOLO('runs/detect/train/weights/best.pt')  
    results = model.predict(source = 'img', conf = 0.25)
    res_plotted = results[0].plot()
    