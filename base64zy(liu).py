
import base64
import json
import os

import cv2
import numpy as np
from PIL import Image

def image_to_base64(img):
    # 将图像转换为JPEG格式
    _, buffer = cv2.imencode('.jpg', img)
    # 将图像数据转换为base64编码的字符串
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str

image_path = r"C:\Users\Jakelee\Desktop\airport_test\02207.jpg"
# image_path = r"C:\Users\Jakelee\Desktop\fks_test\ff.jpg"
image = Image.open(image_path)

image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

PIL_img_dict = {}

PIL_img_dict["Ori_Pic"] = [image_to_base64(image)]
PIL_img_dict["ori_path"] = image_path
PIL_img_dict["Airport_ROI_Pic"] = [image_to_base64(image)]
PIL_img_dict["Airport_ROI_Pos"] = [0, 0, 7900, 4600]
PIL_img_dict["Port_ROI_Pos"] = [0, 0, 7344, 8976]
PIL_img_dict['name'] = "zy"
PIL_img_dict["Port_ROI_Pic"] = [image_to_base64(image)]
print("保存图像完成！")

json_file_path = r"D:\E\Python\flask_all\flask\result_json\test.json"
PIL_img_dict['json_file_path'] = json_file_path
json_data = json.dumps(PIL_img_dict, indent=4)
with open(json_file_path, "w") as file:
    file.write(json_data)