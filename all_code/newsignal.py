# -*- coding: gbk -*-
import base64
import logging
import cv2
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

from flask import Flask, request, jsonify
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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


def savejson2(sum,tile_id,all_results):

    data_list = []
    initial_point = {
        "initial_point": sum[0].get('xy_offset')
    }
    data_list.append(initial_point)
    for j in range(len(sum)):
        cls = sum[j].get('cls')
        if sum[j].get('detection_info') == []:
            continue
        detection_info = sum[j]['longitude']
        box_longitude_info = sum[j]['box_longitude']
        box_info = sum[j]['box_point']
        num_id_info = sum[j]['num_id']
        cls_name_combinate_info = sum[j]['cls_name_combinate']
        conf = round(sum[j].get("conf"), 3)

        # corner = [
        #     [round(xywh[0], 3), round(xywh[1], 3)],  # ���Ͻ�
        #     [round(xywh[0] + xywh[2], 3), round(xywh[1], 3)],  # ���Ͻ�
        #     [round(xywh[0] + xywh[2], 3), round(xywh[1] + xywh[3], 3)],  # ���½�
        #     [round(xywh[0], 3), round(xywh[1] + xywh[3], 3)]  # ���½�
        # ]
        corner = [
            [round(box_longitude_info[0], 5), round(box_longitude_info[1], 5)],  # ���Ͻ�
            [round(box_longitude_info[0] + box_longitude_info[2], 5), round(box_longitude_info[1], 5)],  # ���Ͻ�
            [round(box_longitude_info[0] + box_longitude_info[2], 5),
             round(box_longitude_info[1] + box_longitude_info[3], 5)],  # ���½�
            [round(box_longitude_info[0], 5), round(box_longitude_info[1] + box_longitude_info[3], 5)]  # ���½�
        ]
        center_coordinate = [round((box_info[0].item() + box_info[2].item()) / 2, 3),
                             round((box_info[1].item() + box_info[3].item()) / 2, 3)]

        # masks = data.masks.xy[j].tolist()
        target_width = round(sum[j].get("target_width"), 3)
        target_height = round(sum[j].get("target_height"), 3)      # Pixelnumber = len(masks)
        lslist = {
            "class": cls,
            "conf": conf,
            # "corner": corner,
            "corner": corner,
            "num_id": num_id_info,
            "cls_name_combinate_info": cls_name_combinate_info,
            "coordinate": detection_info,
            "center_coordinate": center_coordinate,
            "target_width": target_width,
            "target_height": target_height,
            # "masks_number": Pixelnumber,
            # "masks": masks,
        }
        data_list.append(lslist)
    postfix = "json"
    json_filename = f"image_{tile_id}.{postfix}"
    filename = sum[0].get('json_output_dir')
    file_path = os.path.join(filename,json_filename)
    json_data = json.dumps(data_list, indent=4, ensure_ascii=False)
    with open(file_path, "w") as file:
        file.write(json_data)

        all_results.append({json_filename: data_list})
    return all_results

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
    names = {0: 'oiltank', 1: 'wharf'}
    # �Ľ����ı�ǩ����
    if weights_path == 'weights/ship50.pt':
        names = {0: 'aircraft_carrier', 1: 'other_ship', 2: 'cargo_ship', 3: 'other_ship', 4: 'other_ship',
                 5: 'other_ship', 6: 'other_ship', 7: 'other_ship', 8: 'destroyer', 9: 'other_ship', 10: 'other_ship',
                 11: 'other_ship', 12: 'other_ship', 13: 'other_ship', 14: 'destroyer', 15: 'destroyer',
                 16: 'other_ship', 17: 'other_ship', 18: 'other_ship', 19: 'other_ship', 20: 'frigate',
                 21: 'other_ship', 22: 'other_ship', 23: 'other_ship', 24: 'other_ship', 25: 'destroyer',
                 26: 'other_ship', 27: 'other_ship', 28: 'other_ship', 29: 'destroyer', 30: 'destroyer',
                 31: 'cargo_ship', 32: 'destroyer', 33: 'other_ship', 34: 'cargo_ship', 35: 'other_ship',
                 36: 'other_ship', 37: 'other_ship', 38: 'patrol_ship', 39: 'other_ship', 40: 'other_ship',
                 41: 'other_ship', 42: 'other_ship', 43: 'other_ship', 44: 'other_ship', 45: 'other_ship',
                 46: 'other_ship', 47: 'other_ship', 48: 'other_ship', 49: 'other_ship', 50: 'other_ship',
                 51: 'frigate', 52: 'other_ship', 53: 'other_ship', 54: 'other_ship', 55: 'other_ship', 56: 'destroyer',
                 57: 'destroyer', 58: 'frigate', 59: 'other_ship', 60: 'frigate', 61: 'other_ship', 62: 'destroyer',
                 63: 'other_ship', 64: 'other_ship', 65: 'other_ship', 66: 'other_ship', 67: 'other_ship',
                 68: 'frigate', 69: 'other_ship', 70: 'other_ship', 71: 'other_ship', 72: 'other_ship', 73: 'destroyer',
                 74: 'other_ship', 75: 'other_ship', 76: 'other_ship', 77: 'destroyer', 78: 'other_ship',
                 79: 'destroyer', 80: 'other_ship', 81: 'other_ship', 82: 'frigate', 83: 'other_ship', 84: 'other_ship',
                 85: 'other_ship', 86: 'other_ship', 87: 'other_ship', 88: 'other_ship', 89: 'other_ship',
                 90: 'other_ship', 91: 'other_ship', 92: 'other_ship', 93: 'other_ship', 94: 'other_ship',
                 95: 'destroyer', 96: 'other_ship', 97: 'destroyer', 98: 'other_ship', 99: 'other_ship',
                 100: 'other_ship', 101: 'other_ship', 102: 'destroyer', 103: 'other_ship', 104: 'other_ship',
                 105: 'other_ship', 106: 'other_ship', 107: 'other_ship', 108: 'other_ship', 109: 'destroyer',
                 110: 'other_ship', 111: 'destroyer', 112: 'other_ship', 113: 'destroyer', 114: 'other_ship',
                 115: 'other_ship', 116: 'other_ship', 117: 'other_ship', 118: 'aircraft_carrier', 119: 'other_ship',
                 120: 'other_ship', 121: 'aircraft_carrier', 122: 'aircraft_carrier', 123: 'other_ship',
                 124: 'other_ship', 125: 'other_ship', 126: 'other_ship', 127: 'other_ship', 128: 'other_ship',
                 129: 'other_ship', 130: 'other_ship', 131: 'other_ship', 132: 'destroyer'}
    elif weights_path == 'weights/airplane.pt':
        names = {0: 'bomber', 1: 'transport_plane', 2: 'warplane', 3: 'transport_plane', 4: 'early_warning_airplane', 5: 'oil_plane', 6: 'antisubmarine_plane'}
    elif weights_path == 'weights\\airport2.pt':
        names = {0: 'TBuild', 1: 'Taxiway', 2: 'Apron', 3: 'Runway', 4: 'Hangar', 5: 'Landpad'}
    elif weights_path == 'weights/harbor.pt':
        names = {0: 'oiltank', 1: 'wharf'}
    category_counts = {key: 0 for key in names}  # ������
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
            if weights_path == 'weights/ship50.pt':
                names = {0: 'aircraft_carrier', 1: 'other_ship', 2: 'cargo_ship', 3: 'other_ship', 4: 'other_ship',
                         5: 'other_ship', 6: 'other_ship', 7: 'other_ship', 8: 'destroyer', 9: 'other_ship',
                         10: 'other_ship', 11: 'other_ship', 12: 'other_ship', 13: 'other_ship', 14: 'destroyer',
                         15: 'destroyer', 16: 'other_ship', 17: 'other_ship', 18: 'other_ship', 19: 'other_ship',
                         20: 'frigate', 21: 'other_ship', 22: 'other_ship', 23: 'other_ship', 24: 'other_ship',
                         25: 'destroyer', 26: 'other_ship', 27: 'other_ship', 28: 'other_ship', 29: 'destroyer',
                         30: 'destroyer', 31: 'cargo_ship', 32: 'destroyer', 33: 'other_ship', 34: 'cargo_ship',
                         35: 'other_ship', 36: 'other_ship', 37: 'other_ship', 38: 'patrol_ship', 39: 'other_ship',
                         40: 'other_ship', 41: 'other_ship', 42: 'other_ship', 43: 'other_ship', 44: 'other_ship',
                         45: 'other_ship', 46: 'other_ship', 47: 'other_ship', 48: 'other_ship', 49: 'other_ship',
                         50: 'other_ship', 51: 'frigate', 52: 'other_ship', 53: 'other_ship', 54: 'other_ship',
                         55: 'other_ship', 56: 'destroyer', 57: 'destroyer', 58: 'frigate', 59: 'other_ship',
                         60: 'frigate', 61: 'other_ship', 62: 'destroyer', 63: 'other_ship', 64: 'other_ship',
                         65: 'other_ship', 66: 'other_ship', 67: 'other_ship', 68: 'frigate', 69: 'other_ship',
                         70: 'other_ship', 71: 'other_ship', 72: 'other_ship', 73: 'destroyer', 74: 'other_ship',
                         75: 'other_ship', 76: 'other_ship', 77: 'destroyer', 78: 'other_ship', 79: 'destroyer',
                         80: 'other_ship', 81: 'other_ship', 82: 'frigate', 83: 'other_ship', 84: 'other_ship',
                         85: 'other_ship', 86: 'other_ship', 87: 'other_ship', 88: 'other_ship', 89: 'other_ship',
                         90: 'other_ship', 91: 'other_ship', 92: 'other_ship', 93: 'other_ship', 94: 'other_ship',
                         95: 'destroyer', 96: 'other_ship', 97: 'destroyer', 98: 'other_ship', 99: 'other_ship',
                         100: 'other_ship', 101: 'other_ship', 102: 'destroyer', 103: 'other_ship', 104: 'other_ship',
                         105: 'other_ship', 106: 'other_ship', 107: 'other_ship', 108: 'other_ship', 109: 'destroyer',
                         110: 'other_ship', 111: 'destroyer', 112: 'other_ship', 113: 'destroyer', 114: 'other_ship',
                         115: 'other_ship', 116: 'other_ship', 117: 'other_ship', 118: 'aircraft_carrier',
                         119: 'other_ship', 120: 'other_ship', 121: 'aircraft_carrier', 122: 'aircraft_carrier',
                         123: 'other_ship', 124: 'other_ship', 125: 'other_ship', 126: 'other_ship', 127: 'other_ship',
                         128: 'other_ship', 129: 'other_ship', 130: 'other_ship', 131: 'other_ship', 132: 'destroyer'}

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
            cls_name_combinate_info = []
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
                        #-----------------------������-----------------------
                        clss = names[cls.item()]
                        category = next(key for key, value in names.items() if value == clss)
                        category_counts[category] += 1
                        cls_name_combinate = clss + '_' + str(category_counts[category])
                        #-----------------------������-----------------------
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
                                'box_point': [x1_longitude_change, y1_longitude_change, x2_longitude_change,
                                              y2_longitude_change],  # ����
                                'box_longitude': [t_x1, t_y1, t_x2, t_y2],
                                'longitude': [str_longitude_x, str_longitude_y],  # ��γ������
                                "num_id": num_id,
                                "xy_offset": [x, y],
                                "json_output_dir": json_output_dir,
                                "cls_name_combinate":cls_name_combinate
                            })
                            box_info.append({"box_longitude": [t_x1,t_y1,t_x2,t_y2],"box_point":[x1_longitude_change,y1_longitude_change,x2_longitude_change,y2_longitude_change] })
                            cls_name_combinate_info.append({"cls_name_combinate": cls_name_combinate})
                    # ���½���ļ����Ϣ
                    result[0].detection_info = detection_info
                    result[0].box_info = box_info
                    result[0].num_info = num_info
                    result[0].cls_name_combinate_info = cls_name_combinate_info
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

                        # -----------------------������-----------------------
                        clss = names[cls.item()]
                        category = next(key for key, value in names.items() if value == clss)
                        category_counts[category] += 1
                        cls_name_combinate = clss + '_' + str(category_counts[category])

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
                            'box_point': [x1_longitude_change, y1_longitude_change, x2_longitude_change,
                                          y2_longitude_change],
                            "cls": clss,
                            'box': det,  # ����
                            'box_longitude': [t_x1,t_y1,t_x2,t_y2],
                            'mask': [msk, x, y],  # ����
                            'longitude': [str_longitude_x, str_longitude_y],  # ��γ������
                            "num_id": num_id,
                            "xy_offset": [x, y],
                            "json_output_dir": json_output_dir,
                            # "detection_info": detection_info,
                            # "box_info": box_info,
                            # "num_info": num_info,
                            'cls_name_combinate':cls_name_combinate
                        })
                        box_info.append({"box_longitude": [t_x1, t_y1, t_x2, t_y2],
                                     "box_point": [x1_longitude_change, y1_longitude_change, x2_longitude_change,
                                                   y2_longitude_change]})
                        cls_name_combinate_info.append({"cls_name_combinate": cls_name_combinate})

                    # ���½���ļ����Ϣ
                    result[0].detection_info = detection_info
                    result[0].box_info = box_info
                    result[0].num_info = num_info
                    result[0].cls_name_combinate_info = cls_name_combinate_info
                    resultall.append(result)



    # # ��ȡ���겢ת��Ϊ����
    coords = torch.stack([item['box'][:4] for item in combined_results])
    # ��ȡ����
    scores = torch.stack([item['box'][4] for item in combined_results])
    # ���зǼ���ֵ����
    keep = nms(coords, scores, iou_threshold=0.5)
    # ��ȡ�Ǽ���ֵ���ƺ�Ľ��
    nms_results = [combined_results[i] for i in keep]
    # nms_results = combined_results

    #-------------------

    data_temp = []
    index_temp = []

    for index, data in enumerate(nms_results):
        if data["cls"] == "Runway":
            data_temp.append(nms_results[index])
            index_temp.append(index)

    r = []

    def merge_runways_list(runways_list, threshold,threshold2):
        merged_runways = []  # ���ڴ�źϲ�����ܵ�
        while len(runways_list) > 0:
            # ���б���ȡ��һ���ܵ�
            runway1 = runways_list.pop(0)
            # �ҳ��뵱ǰ�ܵ����Ϻϲ��������ܵ�
            merge_candidates = [r for r in runways_list if abs(r['box'][0] - runway1['box'][0]) < threshold
                                                 or abs(r['box'][3] - runway1['box'][1]) < threshold2]

            # ����з����������ܵ�������кϲ�
            if merge_candidates:
                # ��ʼ���ϲ�������ܵ���Ϣ
                merged_runway = {
                    'target_width': runway1['target_width'],
                    'target_height': runway1['target_height'],
                    'conf': runway1['conf'],
                    'box_point': runway1['box_point'],
                    'cls': runway1['cls'],
                    'box': [runway1['box'][0], runway1['box'][1], runway1['box'][2], runway1['box'][3]],
                    'box_longitude': [runway1['box_longitude'][0], runway1['box_longitude'][1],
                                      runway1['box_longitude'][2], runway1['box_longitude'][3]],
                    'longitude': [runway1['longitude'][0], runway1['longitude'][1]],
                    'num_id': runway1['num_id'],
                    'xy_offset': runway1['xy_offset'],
                    'json_output_dir': runway1['json_output_dir'],
                    # 'detection_info': [d for d in runway1['detection_info']],
                    # 'num_info': [n for n in runway1['num_info']],
                    # 'box_info': [b for b in runway1['box_info']],
                    'cls_name_combinate': runway1['cls_name_combinate']
                }
                # ��ʼ���ϲ������mask��Ϣ
                merged_mask = [
                    [runway1['mask'][0]],
                    [runway1['mask'][1]],
                    [runway1['mask'][2]]
                ]
                # �������������ܵ�����ϲ��б��������ԭʼ�б����Ƴ�
                for runway2 in merge_candidates:


                    tensor1 = torch.tensor(min(merged_runway['box'][0], runway2['box'][0]), device='cuda:0')
                    tensor2 = torch.tensor(min(merged_runway['box'][1], runway2['box'][1]), device='cuda:0')
                    tensor3 = torch.tensor(max(merged_runway['box'][2], runway2['box'][2]), device='cuda:0')
                    tensor4 = torch.tensor(max(merged_runway['box'][3], runway2['box'][3]), device='cuda:0')
                    tensor5 = torch.tensor(max(merged_runway['conf'], runway2['conf']), device='cuda:0')
                    tensor6 = torch.tensor(3.0, device='cuda:0')

                    # ����Щ�������ӳ�һ������
                    stacked_tensor = torch.cat(
                        [tensor1.unsqueeze(0), tensor2.unsqueeze(0), tensor3.unsqueeze(0), tensor4.unsqueeze(0),
                         tensor5.unsqueeze(0), tensor6.unsqueeze(0)])

                    # �������ֵ�� merged_runway['box']
                    merged_runway['box'] = stacked_tensor

                    # print(merged_runway)
                    merged_runway['box_longitude'] = [
                        min(merged_runway['box_longitude'][0], runway2['box_longitude'][0]),
                        min(merged_runway['box_longitude'][1], runway2['box_longitude'][1]),
                        max(merged_runway['box_longitude'][2], runway2['box_longitude'][2]),
                        max(merged_runway['box_longitude'][3], runway2['box_longitude'][3])
                    ]
                    merged_runway['longitude'] = [
                        (merged_runway['longitude'][0] + runway2['longitude'][0]) / 2,
                        (merged_runway['longitude'][1] + runway2['longitude'][1]) / 2
                    ]
                    merged_runway['num_id'] = min(merged_runway['num_id'], runway2['num_id'])
                    merged_runway['xy_offset'] = runway1['xy_offset']  # xy_offset���ֲ���
                    merged_runway['json_output_dir'] = runway1['json_output_dir']  # json_output_dir���ֲ���
                    merged_runway['box_point'] = [(merged_runway['box_point'][0] + runway2['box_point'][0]) / 2,
                        (merged_runway['box_point'][1] + runway2['box_point'][1]) / 2,
                                                  (merged_runway['box_point'][2] + runway2['box_point'][2]) / 2,
                    (merged_runway['box_point'][3] + runway2['box_point'][3]) / 2,]
                    # merged_runway['detection_info'].extend(runway2['detection_info'])
                    # merged_runway['num_info'].extend(runway2['num_info'])
                    # merged_runway['box_info'].extend(runway2['box_info'])
                    merged_runway['cls_name_combinate'] = runway1['cls_name_combinate']

                    merged_mask[0].append(runway2['mask'][0])
                    merged_mask[1].append(runway2['mask'][1])
                    merged_mask[2].append(runway2['mask'][2])
                    runways_list.remove(runway2)
                # [��һ�����mask��x��y,�ڶ������mask��x��y,���������mask��x��y,...]
                merged_runway['mask'] = [
                     merged_mask[0],
                     merged_mask[1],
                     merged_mask[2]
                ]

                merged_runways.append(merged_runway)
            else:
                # ���û�з����������ܵ����򽫵�ǰ�ܵ�ֱ�Ӽ���ϲ��б�
                merged_runways.append(runway1)
        return merged_runways

    # �ϲ��ܵ�
    threshold = 158
    threshold2 = 50


    merged_runways = merge_runways_list(data_temp,threshold,threshold2)



    # print('merged_runways:',merged_runways)



    # ��ɾ��nms_results�з����������ܵ���Ϣ
    nms_results = [r for r in nms_results if r["cls"] != "Runway"]
    # ���ϲ�����ܵ����뵽nms_results��
    nms_results.extend(merged_runways)
    # print("nms_results:",nms_results)
    # ��nms_results��ӵ�results��
    results.extend(nms_results)




    # all_results.append(results)
    # -------------------
    flag = 0
    while len(nms_results) > 0:
        current_offset = nms_results[0].get("xy_offset")
        sum = [nms_results[0]]
        remove_indices = [0]  # ���ڼ�¼Ҫɾ��������
        for idx, result in enumerate(nms_results[1:], start=1):
            if result.get("xy_offset") == current_offset:
                sum.append(result)
                remove_indices.append(idx)
        # �Ӻ���ǰɾ����������������
        for idx in reversed(remove_indices):
            del nms_results[idx]
        all_results = savejson2(sum, flag,all_results)
        flag += 1

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



import cv2
import numpy as np

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
            # clss = classname[cls.item()]
            clss = box["cls_name_combinate"]
            # ��ȡ����Ӧ����ɫ
            colors = color[int(cls.item())]
            # ��ͼ���ϻ�����
            cv2.rectangle(image, (x1, y1), (x2, y2), colors, thickness=3)
            # ������ǩ�������������Ŷ�
            # label = f"Class: {clss}, Conf: {conf:.2f}"
            num_id = box["num_id"]
            label = f" {clss}"
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
            # label = f"{clss}"
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
            # clss = classname[cls.item()]
            clss = box["cls_name_combinate"]
            # ��ȡ����Ӧ����ɫ
            colors = color[int(cls.item())]
            num_id = box["num_id"]
            # --------------------��ͼ����ǩ--------------------
            label = f"{clss}"
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
            # clss = classname[cls.item()]
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
            label = f"{clss}"
            cv2.putText(big_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
            # big_img = cv2ImgAddTextWithBoxAndLabel(Use_Masks,big_img, x1, y1, x2, y2, label, colors, 80, colors, 2, colors, 2)
            # print("mask[1]:",box["mask"][1])

            if isinstance(box["mask"][1],int):
                # print("--------------1-----------------\n", box["mask"])
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


            else:

                matrix_num = len(box['mask'][1])
                #�Һϲ��������½ǵ�����[max_x,max_y]
                max_x = 0
                max_y = 0
                for i, (mask_, x1_, y1_) in enumerate(zip(box["mask"][0], box["mask"][1], box["mask"][2])):

                    x1_ = x1_  # ��x����Ϊ��ǰ�����x����
                    y1_ = y1_  # ��y����Ϊ��ǰ�����y����
                    x2 = x1_ + crop_size[1]  # �������½�x����
                    y2 = y1_ + crop_size[0]  # �������½�y����
                    max_x = max(max_x, x2)  # �������x����
                    max_y = max(max_y, y2)
                # �������ϲ��󣬾������½������ڴ�ͼ�϶�Ӧ��λ��(max_x,max_y)
                max_x += w_big
                max_y += h_big
                # �Һϲ��������Ͻǵ�����[min_x,min_y]
                min_x, min_y = sys.maxsize,sys.maxsize
                for mask_, x1_, y1_ in zip(box["mask"][0], box["mask"][1], box["mask"][2]):
                    x1_ = x1_  # ��x����Ϊ��ǰ�����x����
                    y1_ = y1_  # ��y����Ϊ��ǰ�����y����
                    min_x = min(min_x, x1_)  # ������Сx����
                    min_y = min(min_y, y1_)  # ������Сy����
                # �������ϲ��󣬾������Ͻ������ڴ�ͼ�϶�Ӧ��λ��(min_x,min_y)
                min_x += w_big
                min_y += h_big


                # print("--------------2-----------------\n", box["mask"][1])


                for i, (masknn, x1nn, y1nn) in enumerate(zip(box["mask"][0],box["mask"][1],box["mask"][2])):
                  #----------------��ͼ-----------------------
                    # ��ȡ��������ڴ�ͼ�ϵ�λ��
                    # mask_, x1_, y1_ = box["mask"][:4]
                    #x1nn_ y1nn_���ڼ�¼ԭ����i���������Ͻǵ�λ�ã���Сͼ�ϻ��õ�
                    x1nn_ = x1nn
                    y1nn_ = y1nn
                    # ��i���������Ͻ������ڴ�ͼ�϶�Ӧ��λ��(x1nn,y1nn)
                    x1nn += w_big
                    y1nn += h_big
                    # ��i��mask��������ת��Ϊnumpy����
                    mask = masknn.data.cpu().numpy()

                    # mask��predict�����õ�resize�Ĵ�С����resize������֮ǰcrop�Ĵ�С��ͬ������Ҫ����resize��ȥ��mask
                    # mask = cv2.resize(mask, new_crop_size, interpolation=cv2.INTER_LINEAR)
                    # ȷ�������ڴ�ͼ�ϵ�λ��
                    # x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
                    # ȷ�����벻������ͼ�ı߽�  ��x2��y2���Ǿ������½��ڴ�ͼ�ϵ�����
                    y2 = y1nn+crop_size[0]
                    x2 = x1nn+crop_size[1]

                    mask = cv2.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)
                    # ��������Ĵ�С����Ӧ��ͼ
                    mask = mask[:y2 - y1nn, :x2 - x1nn]

                    # ����һ��������ͬ����С�Ŀհ�ͼ��������ɫ
                    color_mask = np.zeros((y2 - y1nn, x2 - x1nn, 3), dtype=np.uint8)
                    # ֻ������Ϊ1��λ����ɫ
                    color_mask[mask == 1] = colors
                    # ����ɫ���������ӵ���ͼ������ͼ����
                    mask_array[y1nn:y2, x1nn:x2][mask == 1] = color_mask[mask == 1]
                    # print("succeed")
                    #------------------��ͼ----------------------


                    # -----------------��ͼ------------------
                    # ��i��mask��������ת��Ϊnumpy����
                    masknn_ = masknn.data.cpu().numpy()
                    masknn_ = cv2.resize(masknn_, crop_size, interpolation=cv2.INTER_LINEAR)
                    # (x2nn,y2nn)�������½��ڳ���ͼ������
                    x2_, y2_ = x1nn_ + masknn_.shape[1], y1nn_ + masknn_.shape[0]
                    y2_ = min(y2_, height)
                    x2_ = min(x2_, width)
                    small_mask = masknn_[:y2 - y1nn, :x2 - x1nn]
                    samll_color_mask = np.zeros((y2_ - y1nn_, x2_ - x1nn_, 3), dtype=np.uint8)
                    samll_color_mask[small_mask == 1] = colors
                    small_mask_array[y1nn_:y2_, x1nn_:x2_][small_mask == 1] = samll_color_mask[small_mask == 1]
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

    json_output_dir = os.path.basename(big_img_path).replace(".", "_")+ "_" +image_name + '_json'  # json����·��
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
    # print('big_img_with_boxes:',big_img_with_boxes)
    detect_imge = {
        "ori_detect": image_to_base64(big_img_with_boxes),
        "image_name": "large_pic"
    }
    # print(detect_imge,"aaaaa")
    all_result.append(detect_imge)


    # for i,item in enumerate(all_result):
    #     count = 0
    #     if i == len(all_result)-1:
    #         continue
    #     for item2 in item:
    #         if count == 1:
    #             continue
    #         for item3 in item2:
    #             if count == 1:
    #                 continue
                # �� 'box' ��Ӧ��ֵת��Ϊ NumPy ����
                # del item3['box']
                # if 'mask' in list(item3.keys()):
                #     del item3['mask']

            # count +=1

    # print("result",all_result)
    # print(all_result)
    return jsonify(all_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
