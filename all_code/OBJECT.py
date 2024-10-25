# -*- coding: gbk -*-
import base64
import logging
import cv2
import numpy
import requests
import torch
from flask import Flask
from flask_cors import CORS
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
import sys
from pixels_2_longitude import pixels_2_longitude
from main_run import scene_classification
from osgeo import gdal
from image_transfer import draw_longitude
from PIL import Image, ImageDraw, ImageFont
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)
from PIL import Image, ImageDraw, ImageFont
def cv2ImgAddTextWithBoxAndLabel(Use_Masks,img, x1, y1, x2, y2, label, textColor=(0, 255, 0), textSize=20, boxColor=(0, 0, 255), boxThickness=2, labelColor=(255, 0, 0), labelThickness=2):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Draw rectangle
    if not Use_Masks:
        draw.rectangle([(x1, y1), (x2, y2)],fill =None, outline =boxColor,width =boxThickness)
    # Specify font style
    fontStyle = ImageFont.truetype("STSONG.TTF", textSize, encoding="utf-8")

    # Draw text label
    draw.text((x1, y1 - 10), label, labelColor, font=fontStyle)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def inference(weights_path, img_path, crop_size, overlap_size, savejson_func, json_output_dir, postfix,
              Use_Masks,ds_min_x,ds_max_y,geotransform,projection,cut_pic_lefttop_point,num_id,conf_score):
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)
    # 加载模型
    one_results = []
    combined_results = []
    masks = []  # 用来存储所有掩码
    xy_offsets = []  # 用来存储每个掩码相对于原图的偏移量
    model = YOLO(weights_path)
    # 读取图像
    img = img_path
    height, width, _ = img.shape
    # 切割图像并进行预测
    results = []
    all_results = []
    tile_id = 0
    resultall = []

    # ————————————————————————————————————杨——————————————————————————————#
    x_steps = range(0, width, crop_size[0] - overlap_size)
    y_steps = range(0, height, crop_size[1] - overlap_size)
    for i, x in enumerate(x_steps):
        for j, y in enumerate(y_steps):
            if y + crop_size[1] > height:
                y = height - crop_size[1]
            if x + crop_size[0] > width:
                x = width - crop_size[1]
            cropped_image = img[y:y + crop_size[1], x:x + crop_size[0]]
            result = model.predict(cropped_image, imgsz=640,conf=conf_score)
            names = result[0].names
            # 改舰船的标签名字
            if weights_path == 'weights/ship50.pt':
                names = {0: 'aircraft_carrier', 1: 'other_ship', 2: 'cargo_ship', 3: 'other_ship', 4: 'other_ship', 5: 'other_ship', 6: 'other_ship', 7: 'other_ship', 8: 'destroyer', 9: 'other_ship', 10: 'other_ship', 11: 'other_ship', 12: 'other_ship', 13: 'other_ship', 14: 'destroyer', 15: 'destroyer', 16: 'other_ship', 17: 'other_ship', 18: 'other_ship', 19: 'other_ship', 20: 'frigate', 21: 'other_ship', 22: 'other_ship', 23: 'other_ship', 24: 'other_ship', 25: 'destroyer', 26: 'other_ship', 27: 'other_ship', 28: 'other_ship', 29: 'destroyer', 30: 'destroyer', 31: 'cargo_ship', 32: 'destroyer', 33: 'other_ship', 34: 'cargo_ship', 35: 'other_ship', 36: 'other_ship', 37: 'other_ship', 38: 'patrol_ship', 39: 'other_ship', 40: 'other_ship', 41: 'other_ship', 42: 'other_ship', 43: 'other_ship', 44: 'other_ship', 45: 'other_ship', 46: 'other_ship', 47: 'other_ship', 48: 'other_ship', 49: 'other_ship', 50: 'other_ship', 51: 'frigate', 52: 'other_ship', 53: 'other_ship', 54: 'other_ship', 55: 'other_ship', 56: 'destroyer', 57: 'destroyer', 58: 'frigate', 59: 'other_ship', 60: 'frigate', 61: 'other_ship', 62: 'destroyer', 63: 'other_ship', 64: 'other_ship', 65: 'other_ship', 66: 'other_ship', 67: 'other_ship', 68: 'frigate', 69: 'other_ship', 70: 'other_ship', 71: 'other_ship', 72: 'other_ship', 73: 'destroyer', 74: 'other_ship', 75: 'other_ship', 76: 'other_ship', 77: 'destroyer', 78: 'other_ship', 79: 'destroyer', 80: 'other_ship', 81: 'other_ship', 82: 'frigate', 83: 'other_ship', 84: 'other_ship', 85: 'other_ship', 86: 'other_ship', 87: 'other_ship', 88: 'other_ship', 89: 'other_ship', 90: 'other_ship', 91: 'other_ship', 92: 'other_ship', 93: 'other_ship', 94: 'other_ship', 95: 'destroyer', 96: 'other_ship', 97: 'destroyer', 98: 'other_ship', 99: 'other_ship', 100: 'other_ship', 101: 'other_ship', 102: 'destroyer', 103: 'other_ship', 104: 'other_ship', 105: 'other_ship', 106: 'other_ship', 107: 'other_ship', 108: 'other_ship', 109: 'destroyer', 110: 'other_ship', 111: 'destroyer', 112: 'other_ship', 113: 'destroyer', 114: 'other_ship', 115: 'other_ship', 116: 'other_ship', 117: 'other_ship', 118: 'aircraft_carrier', 119: 'other_ship', 120: 'other_ship', 121: 'aircraft_carrier', 122: 'aircraft_carrier', 123: 'other_ship', 124: 'other_ship', 125: 'other_ship', 126: 'other_ship', 127: 'other_ship', 128: 'other_ship', 129: 'other_ship', 130: 'other_ship', 131: 'other_ship', 132: 'destroyer'}
            # 起始坐标点
            result[0].xy_offset = (x, y)
            # 获取类别名称的索引列表
            idx_list = list(names.keys())
            # 创建颜色对象
            colors = Colors()
            # 获取颜色列表
            colorss = [colors(x, True) for x in idx_list]
            # 转换坐标（使其相对于原图）
            detection_info = []
            box_info = []
            num_info = []
            problem = False

            # 如果结果中有检测框
            if len(result[0].boxes.data) > 0:
                # 如果权重路径是'weights\\oil.pt'，'weights\\airplane.pt'或'weights\\harbor.pt'
                if weights_path == 'weights/airplane.pt' or weights_path == 'weights/harbor.pt':
                    # 遍历检测框
                    for det,xywh in zip(result[0].boxes.data,result[0].boxes.xywh):
                        # 克隆张量，避免原地修改
                        problem = False
                        det = det.clone()
                        x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
                        # 裁剪小图转换坐标（使其相对于检测大块图）
                        det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box
                        # -------------进行后处理操作对油库-------------
                        if names[cls.item()] == 'oiltank':
                            w = x2_box.item()-x1_box.item()
                            h = y2_box.item()-y1_box.item()
                            if w/h > 1.5 or h/w > 1.5:
                                print("油库长宽有问题，进行后处理")
                                problem = True
                        # -------------进行后处理操作对油库-------------
                        # # 添加到结果列表
                        # combined_results.append({
                        #     'box': det,  # 检测框
                        # })
                        # # 转换为numpy数组并添加到检测信息列表
                        # det = det.cpu().detach().numpy().tolist()
                        # detection_info.append({"bbox": [round(num, 3) for num in det[:4]]})
                        '将像素点坐标转化为经纬度坐标 ！！注意 这个地方的坐标应该是最大图的坐标进行经纬度转换'
                        x1_longitude_change = det[0]+cut_pic_lefttop_point[0]
                        y1_longitude_change = det[1]+cut_pic_lefttop_point[1]
                        x2_longitude_change = det[2] + cut_pic_lefttop_point[0]
                        y2_longitude_change = det[3] + cut_pic_lefttop_point[1]
                        str_longitude_x, str_longitude_y,t_x1,t_y1,t_x2,t_y2 = pixels_2_longitude(ds_min_x, ds_max_y, geotransform,
                                                                              projection, x1_longitude_change, y1_longitude_change,
                                                                              x2_longitude_change, y2_longitude_change)
                        num_id = num_id+1#每个框的编号
                        detection_info.append({"Longitude": [str_longitude_x, str_longitude_y]})
                        box_info.append({"box_longitude": [t_x1, t_y1, t_x2, t_y2]})
                        num_info.append({"num_id": num_id})
                        _, _, w, h = xywh.cpu().detach().numpy().tolist()
                        target_width = round(w, 3)
                        target_height = round(h, 3)
                        clss = names[cls.item()]
                        if not problem:
                            combined_results.append({
                                "target_width": target_width,
                                "target_height": target_height,
                                "conf": conf.cpu().detach().numpy().tolist(),
                                "cls": clss,
                                'box': det,  # 检测框
                                'box_longitude': [t_x1, t_y1, t_x2, t_y2],
                                'longitude': [str_longitude_x, str_longitude_y],  # 经纬度坐标
                                "num_id": num_id,
                                "xy_offset": [x, y],
                                "json_output_dir": json_output_dir
                            })

                    # 更新结果的检测信息
                    result[0].detection_info = detection_info
                    result[0].box_info = box_info
                    result[0].num_info = num_info
                    resultall.append(result)
                else:
                    # 遍历检测框和掩码
                    for det, msk, xywh in zip(result[0].boxes.data, result[0].masks.data, result[0].boxes.xywh):
                        # 克隆张量，避免原地修改
                        det = det.clone()
                        x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
                        # 转换坐标（使其相对于原图）
                        det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box

                        # # 添加到结果列表
                        # combined_results.append({
                        #     'box': det,  # 检测框
                        #     'mask': [msk, x, y],  # 掩码
                        # })
                        # # 转换为numpy数组并添加到检测信息列表
                        # det = det.cpu().detach().numpy().tolist()
                        # detection_info.append({"bbox": [round(num, 3) for num in det[:4]]})
                        '将像素点坐标转化为经纬度坐标 ！！注意 这个地方的坐标应该是最大图的坐标进行经纬度转换'
                        x1_longitude_change = det[0] + cut_pic_lefttop_point[0]
                        y1_longitude_change = det[1] + cut_pic_lefttop_point[1]
                        x2_longitude_change = det[2] + cut_pic_lefttop_point[0]
                        y2_longitude_change = det[3] + cut_pic_lefttop_point[1]
                        str_longitude_x,str_longitude_y,t_x1,t_y1,t_x2,t_y2 = pixels_2_longitude(ds_min_x, ds_max_y, geotransform,
                                                                              projection, x1_longitude_change,
                                                                              y1_longitude_change,
                                                                              x2_longitude_change, y2_longitude_change)
                        num_id = num_id + 1  # 每个框的编号
                        detection_info.append({"Longitude": [str_longitude_x, str_longitude_y]})
                        box_info.append({"box_longitude": [t_x1, t_y1, t_x2, t_y2]})
                        num_info.append({"num_id": num_id})
                        _, _, w, h = xywh.cpu().detach().numpy().tolist()
                        target_width = round(w, 3)
                        target_height = round(h, 3)
                        clss = names[cls.item()]
                        combined_results.append({
                            "target_width": target_width,
                            "target_height": target_height,
                            "conf": conf.cpu().detach().numpy().tolist(),
                            "cls": clss,
                            'box': det,  # 检测框
                            'box_longitude': [t_x1,t_y1,t_x2,t_y2],
                            'mask': [msk, x, y],  # 掩码
                            'longitude': [str_longitude_x, str_longitude_y],  # 经纬度坐标
                            "num_id": num_id,
                            "xy_offset": [x, y],
                            "detection_info": detection_info,
                            "box_info": box_info,
                            "num_info": num_info
                        })

                    # 更新结果的检测信息
                    result[0].detection_info = detection_info
                    result[0].box_info = box_info
                    result[0].num_info = num_info
                    resultall.append(result)



    # 提取坐标并转换为张量
    coords = torch.stack([item['box'][:4] for item in combined_results])
    # 提取分数
    scores = torch.stack([item['box'][4] for item in combined_results])
    #类别
    class_run=torch.stack([item['box'][5] for item in combined_results])
    selected_boxes = []

    # 2. 循环遍历combined_results列表
    for item in combined_results:
        # 3. 检查第五列的元素是否等于3
        if item['box'][5] == 3:
            # 4. 添加符合条件的box到列表中
            selected_boxes.append(item['box'][:4].cpu().numpy().tolist())

    print(selected_boxes)
    # 5. 将列表保存为JSON文件
    def save_json(selected_boxes, filename):
        # 如果文件已存在，则添加一个数字后缀，直到找到一个可用的文件名
        count = -1
        while True:
            new_filename = f"json_l/{filename}_{count}.json" if count > 0 else filename
            if not os.path.exists(new_filename):
                break
            count += 1

        with open(new_filename, 'w') as file:
            for index, box in enumerate(selected_boxes):
                json.dump(box, file, indent=4)
                if index < len(selected_boxes) - 1:
                    file.write(',')

        print(f"Saved JSON data to: {new_filename}")
    # save_json_l = 'json_l'
    out_json = 'json_l'

    save_json(selected_boxes,out_json)


    # 进行非极大值抑制
    keep = nms(coords, scores, iou_threshold=0.5)
    # 获取非极大值抑制后的结果
    nms_results = [combined_results[i] for i in keep]
    # 将nms_results添加到results中




    results.extend(nms_results)


    return results, img, colorss, names, all_results, weights_path, num_id


# 定义一个函数来根据类别选择颜色
def get_color_by_class(cls):
    # 这里你可以定义一个颜色映射
    color_map = {
        0: (255, 0, 0),  # 类别0的颜色（例如，红色）
        1: (0, 255, 0),  # 类别1的颜色（例如，绿色）
        # ...为其他类别定义颜色
    }
    return color_map.get(cls, (255, 255, 255))  # 默认颜色为白色


def draw_boxes(cut_pic_lefttop_point,big_img,image, boxes, color, classname, weights_path, crop_size,Use_Masks):
    # 如果权重路径是'weights\\oil.pt'，'weights\\airplane.pt'或'weights\\harbor.pt'
    w_big = cut_pic_lefttop_point[0]
    h_big = cut_pic_lefttop_point[1]
    # ----------------------------------杨----------------------------------------
    # 舰船的类别需要更改标签
    if weights_path == 'weights/ship50.pt':
        classname = {0: 'aircraft_carrier', 1: 'other_ship', 2: 'cargo_ship', 3: 'other_ship', 4: 'other_ship', 5: 'other_ship', 6: 'other_ship', 7: 'other_ship', 8: 'destroyer', 9: 'other_ship', 10: 'other_ship', 11: 'other_ship', 12: 'other_ship', 13: 'other_ship', 14: 'destroyer', 15: 'destroyer', 16: 'other_ship', 17: 'other_ship', 18: 'other_ship', 19: 'other_ship', 20: 'frigate', 21: 'other_ship', 22: 'other_ship', 23: 'other_ship', 24: 'other_ship', 25: 'destroyer', 26: 'other_ship', 27: 'other_ship', 28: 'other_ship', 29: 'destroyer', 30: 'destroyer', 31: 'cargo_ship', 32: 'destroyer', 33: 'other_ship', 34: 'cargo_ship', 35: 'other_ship', 36: 'other_ship', 37: 'other_ship', 38: 'patrol_ship', 39: 'other_ship', 40: 'other_ship', 41: 'other_ship', 42: 'other_ship', 43: 'other_ship', 44: 'other_ship', 45: 'other_ship', 46: 'other_ship', 47: 'other_ship', 48: 'other_ship', 49: 'other_ship', 50: 'other_ship', 51: 'frigate', 52: 'other_ship', 53: 'other_ship', 54: 'other_ship', 55: 'other_ship', 56: 'destroyer', 57: 'destroyer', 58: 'frigate', 59: 'other_ship', 60: 'frigate', 61: 'other_ship', 62: 'destroyer', 63: 'other_ship', 64: 'other_ship', 65: 'other_ship', 66: 'other_ship', 67: 'other_ship', 68: 'frigate', 69: 'other_ship', 70: 'other_ship', 71: 'other_ship', 72: 'other_ship', 73: 'destroyer', 74: 'other_ship', 75: 'other_ship', 76: 'other_ship', 77: 'destroyer', 78: 'other_ship', 79: 'destroyer', 80: 'other_ship', 81: 'other_ship', 82: 'frigate', 83: 'other_ship', 84: 'other_ship', 85: 'other_ship', 86: 'other_ship', 87: 'other_ship', 88: 'other_ship', 89: 'other_ship', 90: 'other_ship', 91: 'other_ship', 92: 'other_ship', 93: 'other_ship', 94: 'other_ship', 95: 'destroyer', 96: 'other_ship', 97: 'destroyer', 98: 'other_ship', 99: 'other_ship', 100: 'other_ship', 101: 'other_ship', 102: 'destroyer', 103: 'other_ship', 104: 'other_ship', 105: 'other_ship', 106: 'other_ship', 107: 'other_ship', 108: 'other_ship', 109: 'destroyer', 110: 'other_ship', 111: 'destroyer', 112: 'other_ship', 113: 'destroyer', 114: 'other_ship', 115: 'other_ship', 116: 'other_ship', 117: 'other_ship', 118: 'aircraft_carrier', 119: 'other_ship', 120: 'other_ship', 121: 'aircraft_carrier', 122: 'aircraft_carrier', 123: 'other_ship', 124: 'other_ship', 125: 'other_ship', 126: 'other_ship', 127: 'other_ship', 128: 'other_ship', 129: 'other_ship', 130: 'other_ship', 131: 'other_ship', 132: 'destroyer'}
    # ----------------------------------杨----------------------------------------
    if not Use_Masks:
        # 遍历所有的框
        for box in boxes:
            # 获取框的坐标，置信度和类别
            x1, y1, x2, y2, conf, cls = box["box"][:6]
            # 将坐标转换为整数
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # 获取类别名称
            clss = classname[cls.item()]
            # 获取类别对应的颜色
            colors = color[int(cls.item())]
            # 在图像上画出框
            cv2.rectangle(image, (x1, y1), (x2, y2), colors, thickness=3)
            # 创建标签，包含类别和置信度
            # label = f"Class: {clss}, Conf: {conf:.2f}"
            num_id = box["num_id"]
            label = f"Class: {clss},ID:{num_id}"
            # 在图像上添加标签
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
            '----将信息画出在大图上----'
            x1 += w_big
            x2 += w_big
            y1 += h_big
            y2 += h_big
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            clss = classname[cls.item()]
            colors = color[int(cls.item())]
            #--------------------
            x_center_longitude, y_center_longitude = box["longitude"][:2]

            # d_x, f_x, m_x = x_center_longitude.split('°')[0], x_center_longitude.split('°')[1].split('′')[0], \
            #                 x_center_longitude.split('°')[1].split('′')[1].split('″')[0]
            # d_y, f_y, m_y = x_center_longitude.split('°')[0], x_center_longitude.split('°')[1].split('′')[0], \
            #                 x_center_longitude.split('°')[1].split('′')[1].split('″')[0]
            # x_center_longitude = d_x + " " + f_x + " " + m_x + " "
            # y_center_longitude = d_y + " " + f_y + " " + m_y + " "
            # label = f"{clss},{longitude}"
            # big_img = cv2ImgAddTextWithBoxAndLabel(Use_Masks,big_img, x1, y1, x2, y2,label, colors, 80, colors, 2, colors, 2)
            #--------------------
            cv2.rectangle(big_img, (x1, y1), (x2, y2), colors, thickness=3)
            label = f"Class: {clss},ID:{num_id}"
            cv2.putText(big_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)

    # 如果权重路径不是上述三个之一
    else:
        # 获取大块图像的高度和宽度
        height, width, _ = image.shape
        # 创建一个和大图同样大小的空白图像，用于绘制掩码
        mask_array = np.zeros_like(big_img)
        # 创建一个和子图同样大小的空白图像，用于绘制掩码
        small_mask_array = np.zeros_like(image)
        # 遍历所有的框
        for box in boxes:
            # 获取框的坐标，置信度和类别
            x1, y1, x2, y2, conf, cls = box["box"][:6]
            # 将坐标转换为整数
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # 获取类别名称
            clss = classname[cls.item()]
            # 获取类别对应的颜色
            colors = color[int(cls.item())]
            num_id = box["num_id"]
            # --------------------子图画标签--------------------
            label = f"Class: {clss},ID:{num_id}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
            # --------------------子图画标签--------------------
            # 裁剪的机场大图加上左上角的坐标
            x1 += w_big
            x2 += w_big
            y1 += h_big
            y2 += h_big
            # 将坐标转换为整数
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # 获取类别名称
            clss = classname[cls.item()]
            # 获取类别对应的颜色
            colors = color[int(cls.item())]

            # 创建标签，包含类别和置信度
            # label = f"Class: {clss}, Conf: {conf:.2f}"
            # # 在图像上添加标签
            x_center_longitude, y_center_longitude = box["longitude"][:2]

            #分开度分秒
            # d_x,f_x,m_x = x_center_longitude.split('°')[0],x_center_longitude.split('°')[1].split('′')[0],x_center_longitude.split('°')[1].split('′')[1].split('″')[0]#11°11′22″
            # d_y, f_y, m_y = x_center_longitude.split('°')[0], x_center_longitude.split('°')[1].split('′')[0],x_center_longitude.split('°')[1].split('′')[1].split('″')[0]
            # x_center_longitude = d_x +" "+f_x+" "+m_x+" "
            # y_center_longitude = d_y + " " + f_y + " " + m_y + " "
            label = f"{clss},ID:{num_id}"
            cv2.putText(big_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
            # big_img = cv2ImgAddTextWithBoxAndLabel(Use_Masks,big_img, x1, y1, x2, y2, label, colors, 80, colors, 2, colors, 2)

            # 获取掩码和其在大图上的位置
            mask, x1, y1 = box["mask"][:4]
            x1 += w_big
            y1 += h_big
            # 将掩码转换为numpy数组
            mask = mask.data.cpu().numpy()
            # mask在predict后设置的resize的大小等于resize，不跟之前crop的大小相同，所以要重新resize回去画mask
            mask = cv2.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)
            # 确定掩码在大图上的位置
            x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
            # 确保掩码不超过大图的边界
            y2 = min(y2, height + h_big)
            x2 = min(x2, width + w_big)
            # 调整掩码的大小以适应大图
            mask = mask[:y2 - y1, :x2 - x1]
            # 创建一个和掩码同样大小的空白图像，用于上色
            color_mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            # 只在掩码为1的位置上色
            color_mask[mask == 1] = colors
            # 将上色后的掩码添加到大图的掩码图像上
            mask_array[y1:y2, x1:x2][mask == 1] = color_mask[mask == 1]
            # -----------------子图------------------
            mask_, x1_, y1_ = box["mask"][:4]
            mask_ = mask_.data.cpu().numpy()
            mask_ = cv2.resize(mask_, crop_size, interpolation=cv2.INTER_LINEAR)
            x2_, y2_ = x1_ + mask_.shape[1], y1_ + mask_.shape[0]
            y2_ = min(y2_, height)
            x2_ = min(x2_, width)
            small_mask = mask_[:y2 - y1, :x2 - x1]
            samll_color_mask = np.zeros((y2_ - y1_, x2_ - x1_, 3), dtype=np.uint8)
            samll_color_mask[small_mask == 1] = colors
            small_mask_array[y1_:y2_, x1_:x2_][small_mask == 1] = samll_color_mask[small_mask == 1]
            # -----------------子图------------------
        # 将掩码图像叠加到大图上
        # 这里使用了加权和来保持背景的可见性，你可以调整权重以满足你的需求
        big_img = cv2.addWeighted(big_img, 1.0, mask_array, 0.5, 0)
        image = cv2.addWeighted(image, 1.0, small_mask_array, 0.5, 0)


    return big_img,image


def savejson(results, file_path, names, all_results, json_filename):
    # 保存所有图片的处理结果

    for i in range(len(results)):
        data = results[i]
        data_list = []
        xy = data.xy_offset
        initial_point = {
            "initial_point": xy
        }
        data_list.append(initial_point)
        for j in range(len(data)):
            cls = names[data.boxes.cls[j].item()]
            if data.detection_info == []:
                continue
            detection_info = data.detection_info[j]['Longitude']
            box_longitude_info = data.box_info[j]['box_longitude']
            num_id_info = data.num_info[j]['num_id']
            conf = round(data.boxes.conf[j].item(), 3)
            xywh = data.boxes.xywh[j].cpu().detach().numpy().tolist()

            # corner = [
            #     [round(xywh[0], 3), round(xywh[1], 3)],  # 左上角
            #     [round(xywh[0] + xywh[2], 3), round(xywh[1], 3)],  # 右上角
            #     [round(xywh[0] + xywh[2], 3), round(xywh[1] + xywh[3], 3)],  # 右下角
            #     [round(xywh[0], 3), round(xywh[1] + xywh[3], 3)]  # 左下角
            # ]
            corner = [
                [round(box_longitude_info[0], 5), round(box_longitude_info[1], 5)],  # 左上角
                [round(box_longitude_info[0] + box_longitude_info[2], 5), round(box_longitude_info[1], 5)],  # 右上角
                [round(box_longitude_info[0] + box_longitude_info[2], 5), round(box_longitude_info[1] + box_longitude_info[3], 5)],  # 右下角
                [round(box_longitude_info[0], 5), round(box_longitude_info[1] + box_longitude_info[3], 5)]  # 左下角
            ]
            # masks = data.masks.xy[j].tolist()
            x, y, width, height = xywh
            target_width = round(width, 3)
            target_height = round(height, 3)
            # Pixelnumber = len(masks)
            lslist = {
                "class": cls,
                "conf": conf,
                # "corner": corner,
                "corner": corner,
                "num_id": num_id_info,
                "coordinate": detection_info,
                "target_width": target_width,
                "target_height": target_height,
                # "masks_number": Pixelnumber,
                # "masks": masks,
            }
            data_list.append(lslist)
        json_data = json.dumps(data_list, indent=4, ensure_ascii=False)
        with open(file_path, "w") as file:
            file.write(json_data)

        all_results.append({json_filename: data_list})

    return all_results


def image_to_base64(img):
    # 将图像转换为JPEG格式
    _, buffer = cv2.imencode('.jpg', img)
    # 将图像数据转换为base64编码的字符串
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def draw_legend(colors, labels, img):
    """
    Draw a legend on the image.

    Args:
    - colors (list of tuple): A list of colors (B, G, R) for each class.
    - labels (list of str): A list of labels for each class.
    - img (numpy array): The image on which to draw the legend.
    """
    image = img
    # Image dimensions
    height, width, _ = img.shape

    # Relative sizes of the legend based on image width
    box_width = int(0.5 * width)
    box_height = int(0.05 * height)
    text_offset = int(0.02 * width)
    right_margin = int(0.02 * width)  # Margin on the right side
    text_size = width / 1000.0  # Adjust this value to change the text size
    text_thickness = int(width / 300)  # Adjust this value to change the text thickness

    # Calculate total height of the legend box
    total_legend_height = len(labels) * (box_height + text_offset)

    # Starting coordinates for the legend (adjusted for center-right with margin)
    x_start = width - box_width - right_margin  # Adjusted for right margin
    y_start = (height - total_legend_height) // 2  # Center vertically

    move_right = int(0.15 * width)  # Adjust this value as needed

    # Adjusted starting x coordinate for the legend (move to the right)
    x_starts = width - box_width - right_margin - move_right

    for idx, (color, label) in enumerate(zip(colors, labels)):
        # Draw the color box
        cv2.rectangle(image,
                      (x_starts, y_start + idx * (box_height + text_offset)),
                      (x_starts + box_width, y_start + idx * (box_height + text_offset) + box_height),
                      color, -1, cv2.LINE_AA)

        # Calculate the new text size
        new_text_size = text_size * 2  # 举例增加文本尺寸

        # Put the label text
        cv2.putText(image, label,
                    (x_start - text_offset - int(0.4 * width),
                     y_start + idx * (box_height + text_offset) + box_height // 2 + text_offset // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, new_text_size, (0, 0, 0), text_thickness)
    return image


def infrence(cut_pic_lefttop_point,big_img,img_path, weights_path, crop_size, overlap_size,
             Use_Mask,ds_min_x,ds_max_y,geotransform,projection,big_img_path,image_name,num_id,conf_score):

    json_output_dir = os.path.basename(big_img_path).replace(".", "_")+ "-" +image_name + '_json'  # json保存路径
    file_name_without_extension = os.path.splitext(os.path.basename(big_img_path))[0]
    svimg = file_name_without_extension+ "-" +image_name + '-output.tif'
    small_name = big_img_path.split('.')[0] + "-" +image_name +'-small.tif'
    postfix = 'json'  # 保存格式txt或json
    Use_Masks = Use_Mask
    # 判断采用mask还是水平框

    results, img, colorss, names, all_Port_results, weights_path,num_id = inference(weights_path, img_path, crop_size,
                                                                             overlap_size,
                                                                             savejson, json_output_dir, postfix,
                                                                             Use_Masks,ds_min_x,ds_max_y,geotransform,projection,cut_pic_lefttop_point,num_id,conf_score)


    if len(results) > 0:
        # merged_results = torch.stack(results)
        big_img_with_boxes, img_with_boxes = draw_boxes(cut_pic_lefttop_point,big_img,img, results, colorss, names, weights_path, crop_size, Use_Masks)
        big_pic_paste = big_img_with_boxes
        # # ----------画5等分经纬度线-----------
        # from image_transfer import draw_longitude
        # draw_longitude(svimg, img_path, img_with_boxes)

        # # ----------画5等分经纬度线-----------
        '----画出小图并保存----'
        cv2.imwrite(small_name, img_with_boxes)
        '----画出大图并保存----'
        big_img_with_boxes = draw_longitude(svimg,big_img_path,big_img_with_boxes)


        # img_str = image_to_base64(img_with_boxes)
        # all_Port_results.append({"image": img_str})
        print(f'已将小图保存在{small_name}')
        print(f'已将大图保存在{svimg}')
        detect_imge = {
            "ori_detect": image_to_base64(img_with_boxes),
            # "Roi_detect": image_to_base64(big_img_with_boxes),
            "image_name": image_name
        }
        all_Port_results.append(detect_imge)

    else:
        print("没有检测到任何目标。")


    return results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id


def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return image_data


def convert_image_data(image_data):
    # 将图像数据转换为numpy数组
    np_array = np.frombuffer(image_data, np.uint8)
    # 使用OpenCV的imdecode函数将numpy数组转换为OpenCV的图像格式
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


# def Rio_report():
#     base_url = 'http://127.0.0.1:5006/classify'
#     # 定义参数
#     params = {
#         'image_path': '',
#         'pic_name': "qq",
#         'num_rows': 32,
#         'weights_type': 1,
#     }
#
#     # 发送 GET 请求
#     response = requests.get(base_url, params=params)
#
#     if response.status_code == 200:
#         data = response.json()
#         print(data)
#         return data



@app.route('/start', methods=['GET', 'POST'])
def class_begin():

    # 处理跨域请求
    if request.method == 'OPTIONS':
        # 处理 OPTIONS 请求
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        return ('', 204, response_headers)



    # 权重路径
    weights_dict = {
        '1': r'weights\airport.pt',
        '2': r'weights\ship50.pt',
        '3': r'weights\airplane.pt',
        '4': r'weights\harbor.pt',
        '5': r'weights\oil.pt',
        '6': r'F:\airport\exp\weights\best.pt'
    }

    # 根据参数选着权重
    # weights_type = request.args.get("weights_type")
    # weights_path = weights_dict[str(weights_type)]  # 权重


    img_path_Port = ['zy_Port_roi_post_processing.tif']
    img_path_Airport = ['qq_Airport_roi_post_processing.tif']


    # 场景分类结果
    # data = Rio_report()
    classify_result_json_path = request.args.get("ori_path")

    # classify_result_json_paths = 'qq.json'
    with open(classify_result_json_path, 'r') as json_file:
        # 使用json.load()加载JSON数据
        all_data = json.load(json_file)


    Airport_Pic = convert_image_data(base64_to_image(all_data["Ori_Pic"][0]))  # 原图
    if all_data['name'] =='qq':
        ori_path = all_data["ori_path"]
        Airport_Pos = all_data["Airport_ROI_Pos"] # 裁剪点位
        Airport_ROI_Pic = all_data["Airport_ROI_Pic"] # 裁剪图像
        Port_ROI_Pic = all_data['Port']
    else:
        ori_path = all_data["ori_path"]
        Airport_Pos = all_data["Airport_ROI_Pos"] # 裁剪点位
        Airport_ROI_Pic = all_data["Airport_ROI_Pic"] # 裁剪图像
        Port_Pos = all_data["Port_ROI_Pos"] # 裁剪点位
        Port_ROI_Pic = all_data["Port_ROI_Pic"] # 裁剪图像



    big_img_path = ori_path #原图地址
    img_path_Airport = Airport_ROI_Pic  # 机场裁剪图地址
    img_path_Port = Port_ROI_Pic  # 港口裁剪图地址
    '------------------接口调用需要修改的地方--------------------'

    '-------------------获取tif格式的大图-----------------------'


    ds = gdal.Open(ori_path)
    if ds is None:
        print(f"无法打开文件: {ori_path}")
        return
    # 获取影像宽度和高度
    ds_width = ds.RasterXSize
    ds_height = ds.RasterYSize
    # 获取影像的地理转换信息和投影信息
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds_min_x = geotransform[0]
    ds_max_y = geotransform[3]
    '-------------------获取tif格式的大图-----------------------'
    # 图片路径
    # img_type = request.args.get("img_type")
    overlap_size = 100  # 重叠区域的大小
    crop_size = (640, 640)  # 裁剪大小
    num_id = 0
    conf_score = 0.2
    #  检测飞机
    all_result = []
    # Use_Mask = True
    # out = infrence(img_path_Port, weights_path, crop_size, overlap_size, Use_Mask)
    big_img = Airport_Pic

    if img_path_Airport:
        # 检测飞机
            Use_Mask = False
            if img_path_Airport:
                image_name = 'Airplane'
                weights_path = r'weights/airplane.pt'
                conf_score = 0.42
                cut_pic_lefttop_point = Airport_Pos  # 裁剪的机场的左上位置坐标
                for img_path in img_path_Airport:#每次放入一个区块进行推理，然后保存推理区块图，并且在大图上画出区块的信息
                    img_path = convert_image_data(base64_to_image(img_path))
                    results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id = infrence(cut_pic_lefttop_point,big_img,img_path, weights_path, crop_size, overlap_size,
                                   Use_Mask,ds_min_x,ds_max_y,geotransform,projection,big_img_path,image_name,num_id,conf_score)
                    all_result.append(all_Port_results)


        # # 检测机场
            Use_Mask = True
            if img_path_Airport:
                image_name = 'Airport'
                weights_path = r"weights\airport2.pt"
                cut_pic_lefttop_point = Airport_Pos  # 裁剪的机场的左上位置坐标
                for img_path in img_path_Airport:
                    img_path = convert_image_data(base64_to_image(img_path))
                    overlap_size = 100  # 重叠区域的大小
                    crop_size = (1024, 1024)
                    #将前面检测的大图放入
                    big_img = big_pic_paste
                    results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id = infrence(cut_pic_lefttop_point,big_img,img_path, weights_path, crop_size, overlap_size,
                                   Use_Mask,ds_min_x,ds_max_y,geotransform,projection,big_img_path,image_name,num_id,conf_score)
                    all_result.append(all_Port_results)

    if img_path_Port:
            # 检测 舰船
            Use_Mask = True
            weights_path = r'weights/ship50.pt'
            image_name = 'ship'
            cut_pic_lefttop_point = Port_Pos  #裁剪的港口的左上位置坐标
            for img_path in img_path_Port:
                img_path = convert_image_data(base64_to_image(img_path))
                big_img = big_pic_paste
                results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id = infrence(cut_pic_lefttop_point, big_img, img_path, weights_path, crop_size, overlap_size,
                               Use_Mask, ds_min_x, ds_max_y, geotransform, projection, big_img_path,image_name,num_id,conf_score)
                all_result.append(all_Port_results)

            # 检测 港口
            Use_Mask = False
            cut_pic_lefttop_point = Port_Pos  # 裁剪的港口的左上位置坐标
            image_name = 'harbor'
            print(num_id, "1111111111111111111111111111111111")
            for img_path in img_path_Port:
                weights_path = r'weights/harbor.pt'
                img_path = convert_image_data(base64_to_image(img_path))
                big_img = big_pic_paste
                results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id = infrence(cut_pic_lefttop_point, big_img, img_path, weights_path, crop_size, overlap_size,
                               Use_Mask, ds_min_x, ds_max_y, geotransform, projection, big_img_path,image_name,num_id,conf_score)
                all_result.append(all_Port_results)
    #画出最终合并的大图
    '----画出大图并保存----'
    big_img_with_boxes = draw_longitude('combinate_pic.tif', ori_path, big_pic_paste)
    detect_imge = {
        "ori_detect": image_to_base64(big_img_with_boxes),
        "image_name": "large_pic"
    }
    # print(detect_imge,"aaaaa")
    all_result.append(detect_imge)


    # print(all_result)
    return jsonify(all_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
