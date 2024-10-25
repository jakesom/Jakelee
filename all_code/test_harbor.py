from ultralytics import YOLO
import os

dir_path = r"D:\E\Python\flask_all\flask\crop_img\harbor+cropped_23_17.png"


# Load a model
model = YOLO(r'C:\Users\Jakelee\Desktop\best_oil.pt')  # load a custom model


# Predict with the model
results = model(dir_path, save=True,conf=0.02)  # predict on an image and save to output_dir
