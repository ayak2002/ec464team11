#/usr/bin/python3
import sys
import os
sys.path.append("c:/users/kasse/dropbox/pc/documents/ec463/senior_design_project/code/yolo_test/venv/lib/site-packages")
from ultralytics import YOLO


# model = YOLO('yolov8_tuned.pt') # load a pre-trained model
model = YOLO('model2.pt') # load a pre-trained model
results = model.predict(source = "0", show = True, conf = 0.35)
print(results)