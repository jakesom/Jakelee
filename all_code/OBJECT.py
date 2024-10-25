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
    # ����ģ��
    one_results = []
    combined_results = []
    masks = []  # �����洢��������
    xy_offsets = []  # �����洢ÿ�����������ԭͼ��ƫ����
    model = YOLO(weights_path)
    # ��ȡͼ��
    img = img_path
    height, width, _ = img.shape
    # �и�ͼ�񲢽���Ԥ��
    results = []
    all_results = []
    tile_id = 0
    resultall = []

    # �����������������������������������������������������������������������������������������������������������������������������������#
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
            # �Ľ����ı�ǩ����
            if weights_path == 'weights/ship50.pt':
                names = {0: 'aircraft_carrier', 1: 'other_ship', 2: 'cargo_ship', 3: 'other_ship', 4: 'other_ship', 5: 'other_ship', 6: 'other_ship', 7: 'other_ship', 8: 'destroyer', 9: 'other_ship', 10: 'other_ship', 11: 'other_ship', 12: 'other_ship', 13: 'other_ship', 14: 'destroyer', 15: 'destroyer', 16: 'other_ship', 17: 'other_ship', 18: 'other_ship', 19: 'other_ship', 20: 'frigate', 21: 'other_ship', 22: 'other_ship', 23: 'other_ship', 24: 'other_ship', 25: 'destroyer', 26: 'other_ship', 27: 'other_ship', 28: 'other_ship', 29: 'destroyer', 30: 'destroyer', 31: 'cargo_ship', 32: 'destroyer', 33: 'other_ship', 34: 'cargo_ship', 35: 'other_ship', 36: 'other_ship', 37: 'other_ship', 38: 'patrol_ship', 39: 'other_ship', 40: 'other_ship', 41: 'other_ship', 42: 'other_ship', 43: 'other_ship', 44: 'other_ship', 45: 'other_ship', 46: 'other_ship', 47: 'other_ship', 48: 'other_ship', 49: 'other_ship', 50: 'other_ship', 51: 'frigate', 52: 'other_ship', 53: 'other_ship', 54: 'other_ship', 55: 'other_ship', 56: 'destroyer', 57: 'destroyer', 58: 'frigate', 59: 'other_ship', 60: 'frigate', 61: 'other_ship', 62: 'destroyer', 63: 'other_ship', 64: 'other_ship', 65: 'other_ship', 66: 'other_ship', 67: 'other_ship', 68: 'frigate', 69: 'other_ship', 70: 'other_ship', 71: 'other_ship', 72: 'other_ship', 73: 'destroyer', 74: 'other_ship', 75: 'other_ship', 76: 'other_ship', 77: 'destroyer', 78: 'other_ship', 79: 'destroyer', 80: 'other_ship', 81: 'other_ship', 82: 'frigate', 83: 'other_ship', 84: 'other_ship', 85: 'other_ship', 86: 'other_ship', 87: 'other_ship', 88: 'other_ship', 89: 'other_ship', 90: 'other_ship', 91: 'other_ship', 92: 'other_ship', 93: 'other_ship', 94: 'other_ship', 95: 'destroyer', 96: 'other_ship', 97: 'destroyer', 98: 'other_ship', 99: 'other_ship', 100: 'other_ship', 101: 'other_ship', 102: 'destroyer', 103: 'other_ship', 104: 'other_ship', 105: 'other_ship', 106: 'other_ship', 107: 'other_ship', 108: 'other_ship', 109: 'destroyer', 110: 'other_ship', 111: 'destroyer', 112: 'other_ship', 113: 'destroyer', 114: 'other_ship', 115: 'other_ship', 116: 'other_ship', 117: 'other_ship', 118: 'aircraft_carrier', 119: 'other_ship', 120: 'other_ship', 121: 'aircraft_carrier', 122: 'aircraft_carrier', 123: 'other_ship', 124: 'other_ship', 125: 'other_ship', 126: 'other_ship', 127: 'other_ship', 128: 'other_ship', 129: 'other_ship', 130: 'other_ship', 131: 'other_ship', 132: 'destroyer'}
            # ��ʼ�����
            result[0].xy_offset = (x, y)
            # ��ȡ������Ƶ������б�
            idx_list = list(names.keys())
            # ������ɫ����
            colors = Colors()
            # ��ȡ��ɫ�б�
            colorss = [colors(x, True) for x in idx_list]
            # ת�����꣨ʹ�������ԭͼ��
            detection_info = []
            box_info = []
            num_info = []
            problem = False

            # ���������м���
            if len(result[0].boxes.data) > 0:
                # ���Ȩ��·����'weights\\oil.pt'��'weights\\airplane.pt'��'weights\\harbor.pt'
                if weights_path == 'weights/airplane.pt' or weights_path == 'weights/harbor.pt':
                    # ��������
                    for det,xywh in zip(result[0].boxes.data,result[0].boxes.xywh):
                        # ��¡����������ԭ���޸�
                        problem = False
                        det = det.clone()
                        x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
                        # �ü�Сͼת�����꣨ʹ������ڼ����ͼ��
                        det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box
                        # -------------���к���������Ϳ�-------------
                        if names[cls.item()] == 'oiltank':
                            w = x2_box.item()-x1_box.item()
                            h = y2_box.item()-y1_box.item()
                            if w/h > 1.5 or h/w > 1.5:
                                print("�Ϳⳤ�������⣬���к���")
                                problem = True
                        # -------------���к���������Ϳ�-------------
                        # # ��ӵ�����б�
                        # combined_results.append({
                        #     'box': det,  # ����
                        # })
                        # # ת��Ϊnumpy���鲢��ӵ������Ϣ�б�
                        # det = det.cpu().detach().numpy().tolist()
                        # detection_info.append({"bbox": [round(num, 3) for num in det[:4]]})
                        '�����ص�����ת��Ϊ��γ������ ����ע�� ����ط�������Ӧ�������ͼ��������о�γ��ת��'
                        x1_longitude_change = det[0]+cut_pic_lefttop_point[0]
                        y1_longitude_change = det[1]+cut_pic_lefttop_point[1]
                        x2_longitude_change = det[2] + cut_pic_lefttop_point[0]
                        y2_longitude_change = det[3] + cut_pic_lefttop_point[1]
                        str_longitude_x, str_longitude_y,t_x1,t_y1,t_x2,t_y2 = pixels_2_longitude(ds_min_x, ds_max_y, geotransform,
                                                                              projection, x1_longitude_change, y1_longitude_change,
                                                                              x2_longitude_change, y2_longitude_change)
                        num_id = num_id+1#ÿ����ı��
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
                                'box': det,  # ����
                                'box_longitude': [t_x1, t_y1, t_x2, t_y2],
                                'longitude': [str_longitude_x, str_longitude_y],  # ��γ������
                                "num_id": num_id,
                                "xy_offset": [x, y],
                                "json_output_dir": json_output_dir
                            })

                    # ���½���ļ����Ϣ
                    result[0].detection_info = detection_info
                    result[0].box_info = box_info
                    result[0].num_info = num_info
                    resultall.append(result)
                else:
                    # �������������
                    for det, msk, xywh in zip(result[0].boxes.data, result[0].masks.data, result[0].boxes.xywh):
                        # ��¡����������ԭ���޸�
                        det = det.clone()
                        x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
                        # ת�����꣨ʹ�������ԭͼ��
                        det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box

                        # # ��ӵ�����б�
                        # combined_results.append({
                        #     'box': det,  # ����
                        #     'mask': [msk, x, y],  # ����
                        # })
                        # # ת��Ϊnumpy���鲢��ӵ������Ϣ�б�
                        # det = det.cpu().detach().numpy().tolist()
                        # detection_info.append({"bbox": [round(num, 3) for num in det[:4]]})
                        '�����ص�����ת��Ϊ��γ������ ����ע�� ����ط�������Ӧ�������ͼ��������о�γ��ת��'
                        x1_longitude_change = det[0] + cut_pic_lefttop_point[0]
                        y1_longitude_change = det[1] + cut_pic_lefttop_point[1]
                        x2_longitude_change = det[2] + cut_pic_lefttop_point[0]
                        y2_longitude_change = det[3] + cut_pic_lefttop_point[1]
                        str_longitude_x,str_longitude_y,t_x1,t_y1,t_x2,t_y2 = pixels_2_longitude(ds_min_x, ds_max_y, geotransform,
                                                                              projection, x1_longitude_change,
                                                                              y1_longitude_change,
                                                                              x2_longitude_change, y2_longitude_change)
                        num_id = num_id + 1  # ÿ����ı��
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
                            'box': det,  # ����
                            'box_longitude': [t_x1,t_y1,t_x2,t_y2],
                            'mask': [msk, x, y],  # ����
                            'longitude': [str_longitude_x, str_longitude_y],  # ��γ������
                            "num_id": num_id,
                            "xy_offset": [x, y],
                            "detection_info": detection_info,
                            "box_info": box_info,
                            "num_info": num_info
                        })

                    # ���½���ļ����Ϣ
                    result[0].detection_info = detection_info
                    result[0].box_info = box_info
                    result[0].num_info = num_info
                    resultall.append(result)



    # ��ȡ���겢ת��Ϊ����
    coords = torch.stack([item['box'][:4] for item in combined_results])
    # ��ȡ����
    scores = torch.stack([item['box'][4] for item in combined_results])
    #���
    class_run=torch.stack([item['box'][5] for item in combined_results])
    selected_boxes = []

    # 2. ѭ������combined_results�б�
    for item in combined_results:
        # 3. �������е�Ԫ���Ƿ����3
        if item['box'][5] == 3:
            # 4. ��ӷ���������box���б���
            selected_boxes.append(item['box'][:4].cpu().numpy().tolist())

    print(selected_boxes)
    # 5. ���б���ΪJSON�ļ�
    def save_json(selected_boxes, filename):
        # ����ļ��Ѵ��ڣ������һ�����ֺ�׺��ֱ���ҵ�һ�����õ��ļ���
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


    # ���зǼ���ֵ����
    keep = nms(coords, scores, iou_threshold=0.5)
    # ��ȡ�Ǽ���ֵ���ƺ�Ľ��
    nms_results = [combined_results[i] for i in keep]
    # ��nms_results��ӵ�results��




    results.extend(nms_results)


    return results, img, colorss, names, all_results, weights_path, num_id


# ����һ���������������ѡ����ɫ
def get_color_by_class(cls):
    # ��������Զ���һ����ɫӳ��
    color_map = {
        0: (255, 0, 0),  # ���0����ɫ�����磬��ɫ��
        1: (0, 255, 0),  # ���1����ɫ�����磬��ɫ��
        # ...Ϊ�����������ɫ
    }
    return color_map.get(cls, (255, 255, 255))  # Ĭ����ɫΪ��ɫ


def draw_boxes(cut_pic_lefttop_point,big_img,image, boxes, color, classname, weights_path, crop_size,Use_Masks):
    # ���Ȩ��·����'weights\\oil.pt'��'weights\\airplane.pt'��'weights\\harbor.pt'
    w_big = cut_pic_lefttop_point[0]
    h_big = cut_pic_lefttop_point[1]
    # ----------------------------------��----------------------------------------
    # �����������Ҫ���ı�ǩ
    if weights_path == 'weights/ship50.pt':
        classname = {0: 'aircraft_carrier', 1: 'other_ship', 2: 'cargo_ship', 3: 'other_ship', 4: 'other_ship', 5: 'other_ship', 6: 'other_ship', 7: 'other_ship', 8: 'destroyer', 9: 'other_ship', 10: 'other_ship', 11: 'other_ship', 12: 'other_ship', 13: 'other_ship', 14: 'destroyer', 15: 'destroyer', 16: 'other_ship', 17: 'other_ship', 18: 'other_ship', 19: 'other_ship', 20: 'frigate', 21: 'other_ship', 22: 'other_ship', 23: 'other_ship', 24: 'other_ship', 25: 'destroyer', 26: 'other_ship', 27: 'other_ship', 28: 'other_ship', 29: 'destroyer', 30: 'destroyer', 31: 'cargo_ship', 32: 'destroyer', 33: 'other_ship', 34: 'cargo_ship', 35: 'other_ship', 36: 'other_ship', 37: 'other_ship', 38: 'patrol_ship', 39: 'other_ship', 40: 'other_ship', 41: 'other_ship', 42: 'other_ship', 43: 'other_ship', 44: 'other_ship', 45: 'other_ship', 46: 'other_ship', 47: 'other_ship', 48: 'other_ship', 49: 'other_ship', 50: 'other_ship', 51: 'frigate', 52: 'other_ship', 53: 'other_ship', 54: 'other_ship', 55: 'other_ship', 56: 'destroyer', 57: 'destroyer', 58: 'frigate', 59: 'other_ship', 60: 'frigate', 61: 'other_ship', 62: 'destroyer', 63: 'other_ship', 64: 'other_ship', 65: 'other_ship', 66: 'other_ship', 67: 'other_ship', 68: 'frigate', 69: 'other_ship', 70: 'other_ship', 71: 'other_ship', 72: 'other_ship', 73: 'destroyer', 74: 'other_ship', 75: 'other_ship', 76: 'other_ship', 77: 'destroyer', 78: 'other_ship', 79: 'destroyer', 80: 'other_ship', 81: 'other_ship', 82: 'frigate', 83: 'other_ship', 84: 'other_ship', 85: 'other_ship', 86: 'other_ship', 87: 'other_ship', 88: 'other_ship', 89: 'other_ship', 90: 'other_ship', 91: 'other_ship', 92: 'other_ship', 93: 'other_ship', 94: 'other_ship', 95: 'destroyer', 96: 'other_ship', 97: 'destroyer', 98: 'other_ship', 99: 'other_ship', 100: 'other_ship', 101: 'other_ship', 102: 'destroyer', 103: 'other_ship', 104: 'other_ship', 105: 'other_ship', 106: 'other_ship', 107: 'other_ship', 108: 'other_ship', 109: 'destroyer', 110: 'other_ship', 111: 'destroyer', 112: 'other_ship', 113: 'destroyer', 114: 'other_ship', 115: 'other_ship', 116: 'other_ship', 117: 'other_ship', 118: 'aircraft_carrier', 119: 'other_ship', 120: 'other_ship', 121: 'aircraft_carrier', 122: 'aircraft_carrier', 123: 'other_ship', 124: 'other_ship', 125: 'other_ship', 126: 'other_ship', 127: 'other_ship', 128: 'other_ship', 129: 'other_ship', 130: 'other_ship', 131: 'other_ship', 132: 'destroyer'}
    # ----------------------------------��----------------------------------------
    if not Use_Masks:
        # �������еĿ�
        for box in boxes:
            # ��ȡ������꣬���ŶȺ����
            x1, y1, x2, y2, conf, cls = box["box"][:6]
            # ������ת��Ϊ����
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # ��ȡ�������
            clss = classname[cls.item()]
            # ��ȡ����Ӧ����ɫ
            colors = color[int(cls.item())]
            # ��ͼ���ϻ�����
            cv2.rectangle(image, (x1, y1), (x2, y2), colors, thickness=3)
            # ������ǩ�������������Ŷ�
            # label = f"Class: {clss}, Conf: {conf:.2f}"
            num_id = box["num_id"]
            label = f"Class: {clss},ID:{num_id}"
            # ��ͼ������ӱ�ǩ
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
            '----����Ϣ�����ڴ�ͼ��----'
            x1 += w_big
            x2 += w_big
            y1 += h_big
            y2 += h_big
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            clss = classname[cls.item()]
            colors = color[int(cls.item())]
            #--------------------
            x_center_longitude, y_center_longitude = box["longitude"][:2]

            # d_x, f_x, m_x = x_center_longitude.split('��')[0], x_center_longitude.split('��')[1].split('��')[0], \
            #                 x_center_longitude.split('��')[1].split('��')[1].split('��')[0]
            # d_y, f_y, m_y = x_center_longitude.split('��')[0], x_center_longitude.split('��')[1].split('��')[0], \
            #                 x_center_longitude.split('��')[1].split('��')[1].split('��')[0]
            # x_center_longitude = d_x + " " + f_x + " " + m_x + " "
            # y_center_longitude = d_y + " " + f_y + " " + m_y + " "
            # label = f"{clss},{longitude}"
            # big_img = cv2ImgAddTextWithBoxAndLabel(Use_Masks,big_img, x1, y1, x2, y2,label, colors, 80, colors, 2, colors, 2)
            #--------------------
            cv2.rectangle(big_img, (x1, y1), (x2, y2), colors, thickness=3)
            label = f"Class: {clss},ID:{num_id}"
            cv2.putText(big_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)

    # ���Ȩ��·��������������֮һ
    else:
        # ��ȡ���ͼ��ĸ߶ȺͿ��
        height, width, _ = image.shape
        # ����һ���ʹ�ͼͬ����С�Ŀհ�ͼ�����ڻ�������
        mask_array = np.zeros_like(big_img)
        # ����һ������ͼͬ����С�Ŀհ�ͼ�����ڻ�������
        small_mask_array = np.zeros_like(image)
        # �������еĿ�
        for box in boxes:
            # ��ȡ������꣬���ŶȺ����
            x1, y1, x2, y2, conf, cls = box["box"][:6]
            # ������ת��Ϊ����
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # ��ȡ�������
            clss = classname[cls.item()]
            # ��ȡ����Ӧ����ɫ
            colors = color[int(cls.item())]
            num_id = box["num_id"]
            # --------------------��ͼ����ǩ--------------------
            label = f"Class: {clss},ID:{num_id}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
            # --------------------��ͼ����ǩ--------------------
            # �ü��Ļ�����ͼ�������Ͻǵ�����
            x1 += w_big
            x2 += w_big
            y1 += h_big
            y2 += h_big
            # ������ת��Ϊ����
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # ��ȡ�������
            clss = classname[cls.item()]
            # ��ȡ����Ӧ����ɫ
            colors = color[int(cls.item())]

            # ������ǩ�������������Ŷ�
            # label = f"Class: {clss}, Conf: {conf:.2f}"
            # # ��ͼ������ӱ�ǩ
            x_center_longitude, y_center_longitude = box["longitude"][:2]

            #�ֿ��ȷ���
            # d_x,f_x,m_x = x_center_longitude.split('��')[0],x_center_longitude.split('��')[1].split('��')[0],x_center_longitude.split('��')[1].split('��')[1].split('��')[0]#11��11��22��
            # d_y, f_y, m_y = x_center_longitude.split('��')[0], x_center_longitude.split('��')[1].split('��')[0],x_center_longitude.split('��')[1].split('��')[1].split('��')[0]
            # x_center_longitude = d_x +" "+f_x+" "+m_x+" "
            # y_center_longitude = d_y + " " + f_y + " " + m_y + " "
            label = f"{clss},ID:{num_id}"
            cv2.putText(big_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
            # big_img = cv2ImgAddTextWithBoxAndLabel(Use_Masks,big_img, x1, y1, x2, y2, label, colors, 80, colors, 2, colors, 2)

            # ��ȡ��������ڴ�ͼ�ϵ�λ��
            mask, x1, y1 = box["mask"][:4]
            x1 += w_big
            y1 += h_big
            # ������ת��Ϊnumpy����
            mask = mask.data.cpu().numpy()
            # mask��predict�����õ�resize�Ĵ�С����resize������֮ǰcrop�Ĵ�С��ͬ������Ҫ����resize��ȥ��mask
            mask = cv2.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)
            # ȷ�������ڴ�ͼ�ϵ�λ��
            x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
            # ȷ�����벻������ͼ�ı߽�
            y2 = min(y2, height + h_big)
            x2 = min(x2, width + w_big)
            # ��������Ĵ�С����Ӧ��ͼ
            mask = mask[:y2 - y1, :x2 - x1]
            # ����һ��������ͬ����С�Ŀհ�ͼ��������ɫ
            color_mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            # ֻ������Ϊ1��λ����ɫ
            color_mask[mask == 1] = colors
            # ����ɫ���������ӵ���ͼ������ͼ����
            mask_array[y1:y2, x1:x2][mask == 1] = color_mask[mask == 1]
            # -----------------��ͼ------------------
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
            # -----------------��ͼ------------------
        # ������ͼ����ӵ���ͼ��
        # ����ʹ���˼�Ȩ�������ֱ����Ŀɼ��ԣ�����Ե���Ȩ���������������
        big_img = cv2.addWeighted(big_img, 1.0, mask_array, 0.5, 0)
        image = cv2.addWeighted(image, 1.0, small_mask_array, 0.5, 0)


    return big_img,image


def savejson(results, file_path, names, all_results, json_filename):
    # ��������ͼƬ�Ĵ�����

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
            #     [round(xywh[0], 3), round(xywh[1], 3)],  # ���Ͻ�
            #     [round(xywh[0] + xywh[2], 3), round(xywh[1], 3)],  # ���Ͻ�
            #     [round(xywh[0] + xywh[2], 3), round(xywh[1] + xywh[3], 3)],  # ���½�
            #     [round(xywh[0], 3), round(xywh[1] + xywh[3], 3)]  # ���½�
            # ]
            corner = [
                [round(box_longitude_info[0], 5), round(box_longitude_info[1], 5)],  # ���Ͻ�
                [round(box_longitude_info[0] + box_longitude_info[2], 5), round(box_longitude_info[1], 5)],  # ���Ͻ�
                [round(box_longitude_info[0] + box_longitude_info[2], 5), round(box_longitude_info[1] + box_longitude_info[3], 5)],  # ���½�
                [round(box_longitude_info[0], 5), round(box_longitude_info[1] + box_longitude_info[3], 5)]  # ���½�
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
    # ��ͼ��ת��ΪJPEG��ʽ
    _, buffer = cv2.imencode('.jpg', img)
    # ��ͼ������ת��Ϊbase64������ַ���
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
        new_text_size = text_size * 2  # ���������ı��ߴ�

        # Put the label text
        cv2.putText(image, label,
                    (x_start - text_offset - int(0.4 * width),
                     y_start + idx * (box_height + text_offset) + box_height // 2 + text_offset // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, new_text_size, (0, 0, 0), text_thickness)
    return image


def infrence(cut_pic_lefttop_point,big_img,img_path, weights_path, crop_size, overlap_size,
             Use_Mask,ds_min_x,ds_max_y,geotransform,projection,big_img_path,image_name,num_id,conf_score):

    json_output_dir = os.path.basename(big_img_path).replace(".", "_")+ "-" +image_name + '_json'  # json����·��
    file_name_without_extension = os.path.splitext(os.path.basename(big_img_path))[0]
    svimg = file_name_without_extension+ "-" +image_name + '-output.tif'
    small_name = big_img_path.split('.')[0] + "-" +image_name +'-small.tif'
    postfix = 'json'  # �����ʽtxt��json
    Use_Masks = Use_Mask
    # �жϲ���mask����ˮƽ��

    results, img, colorss, names, all_Port_results, weights_path,num_id = inference(weights_path, img_path, crop_size,
                                                                             overlap_size,
                                                                             savejson, json_output_dir, postfix,
                                                                             Use_Masks,ds_min_x,ds_max_y,geotransform,projection,cut_pic_lefttop_point,num_id,conf_score)


    if len(results) > 0:
        # merged_results = torch.stack(results)
        big_img_with_boxes, img_with_boxes = draw_boxes(cut_pic_lefttop_point,big_img,img, results, colorss, names, weights_path, crop_size, Use_Masks)
        big_pic_paste = big_img_with_boxes
        # # ----------��5�ȷ־�γ����-----------
        # from image_transfer import draw_longitude
        # draw_longitude(svimg, img_path, img_with_boxes)

        # # ----------��5�ȷ־�γ����-----------
        '----����Сͼ������----'
        cv2.imwrite(small_name, img_with_boxes)
        '----������ͼ������----'
        big_img_with_boxes = draw_longitude(svimg,big_img_path,big_img_with_boxes)


        # img_str = image_to_base64(img_with_boxes)
        # all_Port_results.append({"image": img_str})
        print(f'�ѽ�Сͼ������{small_name}')
        print(f'�ѽ���ͼ������{svimg}')
        detect_imge = {
            "ori_detect": image_to_base64(img_with_boxes),
            # "Roi_detect": image_to_base64(big_img_with_boxes),
            "image_name": image_name
        }
        all_Port_results.append(detect_imge)

    else:
        print("û�м�⵽�κ�Ŀ�ꡣ")


    return results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id


def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return image_data


def convert_image_data(image_data):
    # ��ͼ������ת��Ϊnumpy����
    np_array = np.frombuffer(image_data, np.uint8)
    # ʹ��OpenCV��imdecode������numpy����ת��ΪOpenCV��ͼ���ʽ
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


# def Rio_report():
#     base_url = 'http://127.0.0.1:5006/classify'
#     # �������
#     params = {
#         'image_path': '',
#         'pic_name': "qq",
#         'num_rows': 32,
#         'weights_type': 1,
#     }
#
#     # ���� GET ����
#     response = requests.get(base_url, params=params)
#
#     if response.status_code == 200:
#         data = response.json()
#         print(data)
#         return data



@app.route('/start', methods=['GET', 'POST'])
def class_begin():

    # �����������
    if request.method == 'OPTIONS':
        # ���� OPTIONS ����
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        return ('', 204, response_headers)



    # Ȩ��·��
    weights_dict = {
        '1': r'weights\airport.pt',
        '2': r'weights\ship50.pt',
        '3': r'weights\airplane.pt',
        '4': r'weights\harbor.pt',
        '5': r'weights\oil.pt',
        '6': r'F:\airport\exp\weights\best.pt'
    }

    # ���ݲ���ѡ��Ȩ��
    # weights_type = request.args.get("weights_type")
    # weights_path = weights_dict[str(weights_type)]  # Ȩ��


    img_path_Port = ['zy_Port_roi_post_processing.tif']
    img_path_Airport = ['qq_Airport_roi_post_processing.tif']


    # ����������
    # data = Rio_report()
    classify_result_json_path = request.args.get("ori_path")

    # classify_result_json_paths = 'qq.json'
    with open(classify_result_json_path, 'r') as json_file:
        # ʹ��json.load()����JSON����
        all_data = json.load(json_file)


    Airport_Pic = convert_image_data(base64_to_image(all_data["Ori_Pic"][0]))  # ԭͼ
    if all_data['name'] =='qq':
        ori_path = all_data["ori_path"]
        Airport_Pos = all_data["Airport_ROI_Pos"] # �ü���λ
        Airport_ROI_Pic = all_data["Airport_ROI_Pic"] # �ü�ͼ��
        Port_ROI_Pic = all_data['Port']
    else:
        ori_path = all_data["ori_path"]
        Airport_Pos = all_data["Airport_ROI_Pos"] # �ü���λ
        Airport_ROI_Pic = all_data["Airport_ROI_Pic"] # �ü�ͼ��
        Port_Pos = all_data["Port_ROI_Pos"] # �ü���λ
        Port_ROI_Pic = all_data["Port_ROI_Pic"] # �ü�ͼ��



    big_img_path = ori_path #ԭͼ��ַ
    img_path_Airport = Airport_ROI_Pic  # �����ü�ͼ��ַ
    img_path_Port = Port_ROI_Pic  # �ۿڲü�ͼ��ַ
    '------------------�ӿڵ�����Ҫ�޸ĵĵط�--------------------'

    '-------------------��ȡtif��ʽ�Ĵ�ͼ-----------------------'


    ds = gdal.Open(ori_path)
    if ds is None:
        print(f"�޷����ļ�: {ori_path}")
        return
    # ��ȡӰ���Ⱥ͸߶�
    ds_width = ds.RasterXSize
    ds_height = ds.RasterYSize
    # ��ȡӰ��ĵ���ת����Ϣ��ͶӰ��Ϣ
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds_min_x = geotransform[0]
    ds_max_y = geotransform[3]
    '-------------------��ȡtif��ʽ�Ĵ�ͼ-----------------------'
    # ͼƬ·��
    # img_type = request.args.get("img_type")
    overlap_size = 100  # �ص�����Ĵ�С
    crop_size = (640, 640)  # �ü���С
    num_id = 0
    conf_score = 0.2
    #  ���ɻ�
    all_result = []
    # Use_Mask = True
    # out = infrence(img_path_Port, weights_path, crop_size, overlap_size, Use_Mask)
    big_img = Airport_Pic

    if img_path_Airport:
        # ���ɻ�
            Use_Mask = False
            if img_path_Airport:
                image_name = 'Airplane'
                weights_path = r'weights/airplane.pt'
                conf_score = 0.42
                cut_pic_lefttop_point = Airport_Pos  # �ü��Ļ���������λ������
                for img_path in img_path_Airport:#ÿ�η���һ�������������Ȼ�󱣴���������ͼ�������ڴ�ͼ�ϻ����������Ϣ
                    img_path = convert_image_data(base64_to_image(img_path))
                    results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id = infrence(cut_pic_lefttop_point,big_img,img_path, weights_path, crop_size, overlap_size,
                                   Use_Mask,ds_min_x,ds_max_y,geotransform,projection,big_img_path,image_name,num_id,conf_score)
                    all_result.append(all_Port_results)


        # # ������
            Use_Mask = True
            if img_path_Airport:
                image_name = 'Airport'
                weights_path = r"weights\airport2.pt"
                cut_pic_lefttop_point = Airport_Pos  # �ü��Ļ���������λ������
                for img_path in img_path_Airport:
                    img_path = convert_image_data(base64_to_image(img_path))
                    overlap_size = 100  # �ص�����Ĵ�С
                    crop_size = (1024, 1024)
                    #��ǰ����Ĵ�ͼ����
                    big_img = big_pic_paste
                    results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id = infrence(cut_pic_lefttop_point,big_img,img_path, weights_path, crop_size, overlap_size,
                                   Use_Mask,ds_min_x,ds_max_y,geotransform,projection,big_img_path,image_name,num_id,conf_score)
                    all_result.append(all_Port_results)

    if img_path_Port:
            # ��� ����
            Use_Mask = True
            weights_path = r'weights/ship50.pt'
            image_name = 'ship'
            cut_pic_lefttop_point = Port_Pos  #�ü��ĸۿڵ�����λ������
            for img_path in img_path_Port:
                img_path = convert_image_data(base64_to_image(img_path))
                big_img = big_pic_paste
                results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id = infrence(cut_pic_lefttop_point, big_img, img_path, weights_path, crop_size, overlap_size,
                               Use_Mask, ds_min_x, ds_max_y, geotransform, projection, big_img_path,image_name,num_id,conf_score)
                all_result.append(all_Port_results)

            # ��� �ۿ�
            Use_Mask = False
            cut_pic_lefttop_point = Port_Pos  # �ü��ĸۿڵ�����λ������
            image_name = 'harbor'
            print(num_id, "1111111111111111111111111111111111")
            for img_path in img_path_Port:
                weights_path = r'weights/harbor.pt'
                img_path = convert_image_data(base64_to_image(img_path))
                big_img = big_pic_paste
                results, img, colorss, names, all_Port_results, weights_path,big_pic_paste,num_id = infrence(cut_pic_lefttop_point, big_img, img_path, weights_path, crop_size, overlap_size,
                               Use_Mask, ds_min_x, ds_max_y, geotransform, projection, big_img_path,image_name,num_id,conf_score)
                all_result.append(all_Port_results)
    #�������պϲ��Ĵ�ͼ
    '----������ͼ������----'
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
