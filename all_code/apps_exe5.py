# -*- coding: utf-8 -*-
import base64
import logging
import sys

import cv2
import numpy as np
import torch
from flask import Flask
from flask_cors import CORS
from osgeo import gdal
from torchvision.ops import nms
from ultralytics import YOLO

from main_run import scene_classification
from pixels_2_longitude import pixels_2_longitude

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)

from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)


def merge_boxes(boxes):
    # print(boxes)
    boxes = sorted(boxes, key=lambda x: x[6], reverse=True)
    merged_boxes = []
    while len(boxes) > 0:
        current_box = boxes[0]
        del boxes[0]

        overlapping_boxes = []
        for box in boxes:
            iou = compute_iou(current_box, box)
            if iou >= 0.1:
                overlapping_boxes.append(box)

        if len(overlapping_boxes) > 0:
            merged_box = list(current_box)
            for box in overlapping_boxes:
                merged_box[0] = min(merged_box[0], box[0])
                merged_box[1] = min(merged_box[1], box[1])
                merged_box[2] = max(merged_box[2], box[2])
                merged_box[3] = max(merged_box[3], box[3])
                if box[6] > merged_box[6]:
                    merged_box[5] = box[5]
                    merged_box[6] = box[6]
            for box in overlapping_boxes:
                boxes.remove(box)
            merged_boxes.append(merged_box)
        else:
            merged_boxes.append(current_box)
    return merged_boxes


def compute_iou(box1, box2):
    # 计算两个框的重叠面积比例（IoU）
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection / float(area1 + area2 - intersection)
    return iou


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


def inference(weights_path, img_path, crop_size, overlap_size, savejson_func, json_output_dir, postfix, Use_Mask):
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)
    # 加载模型
    one_results = []
    combined_results = []
    masks = []  # 用来存储所有掩码
    xy_offsets = []  # 用来存储每个掩码相对于原图的偏移量
    model = YOLO(weights_path)
    # 读取图像
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    # ——————————经纬度坐标转换————————————
    ds = gdal.Open(img_path)
    if ds is None:
        print(f"无法打开文件: {img_path}")
        return
    # 获取影像宽度和高度
    ds_width = ds.RasterXSize
    height = ds.RasterYSize
    # 获取影像的地理转换信息和投影信息
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds_min_x = geotransform[0]
    ds_max_y = geotransform[3]
    # ——————————经纬度坐标转换————————————
    # 切割图像并进行预测
    results = []
    all_results = []
    tile_id = 0
    weight_all = [r'weights\harbor.pt', r'weights\bridge.pt', r'weights\oil.pt']
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
            result = model.predict(cropped_image, agnostic_nms=True, conf=0.5)
            names = result[0].names
            # 起始坐标点
            result[0].xy_offset = (x, y)
            idx_list = list(names.keys())
            colors = Colors()
            colorss = [colors(x, True) for x in idx_list]
            # 转换坐标（使其相对于原图）
            detection_info = []
            # if len(result[0].boxes.data) > 0:
            #     for det in result[0].boxes.data:
            #         det = det.clone()  # 克隆张量，避免原地修改
            #         x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
            #         det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box
            #         combined_results.append({
            #             'box': det,  # 检测框
            #             'mask': [result[0].masks, x, y,cls],  # 掩码
            #         })
            #         det = det.cpu().detach().numpy().tolist()
            #         detection_info.append({"bbox": [round(num, 3) for num in det[:4]]})
            if len(result[0].boxes.data) > 0:
                if not Use_Mask:
                    for det in result[0].boxes.data:
                        det = det.clone()  # 克隆张量，避免原地修改
                        x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
                        det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box
                        # longitude = det.clone()
                        str_longitude_x, str_longitude_y = pixels_2_longitude(ds_min_x, ds_max_y, geotransform,
                                                                              projection, x + x1_box, y + y1_box,
                                                                              x + x2_box, y + y2_box)
                        combined_results.append({
                            'box': det,  # 检测框
                            'longitude': [str_longitude_x, str_longitude_y]  # 经纬度坐标
                        })
                        detection_info.append({"Longitude": [str_longitude_x, str_longitude_y]})
                    result[0].detection_info = detection_info
                    print("transfored")
                else:
                    for det, msk in zip(result[0].boxes.data, result[0].masks.data):
                        det = det.clone()  # 克隆张量，避免原地修改
                        x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
                        det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box
                        combined_results.append({
                            'box': det,  # 检测框
                            'mask': [msk, x, y],  # 掩码
                        })
                        det = det.cpu().detach().numpy().tolist()
                        detection_info.append({"bbox": [round(num, 3) for num in det[:4]]})
                    result[0].detection_info = detection_info
            if savejson_func is not None:
                json_filename = f"image_{tile_id}.{postfix}"
                json_file_path = os.path.join(json_output_dir, json_filename)
                all_results = savejson_func(result, json_file_path, names, all_results, json_filename)
                tile_id += 1
    coords = torch.stack([item['box'][:4] for item in combined_results])  # 提取坐标并转换为张量
    scores = torch.stack([item['box'][4] for item in combined_results])
    keep = nms(coords, scores, iou_threshold=0.5)  # NMS

    nms_results = [combined_results[i] for i in keep]
    # # 将nms_results添加到results中
    results.extend(nms_results)
    return results, img, colorss, names, all_results, weights_path


def draw_boxes(image, boxes, color, classname, weights_path, crop_size, Use_Mask):
    # w_big, h_big = 33, 6314
    # w_big, h_big = 13681, 8625
    # w_big, h_big = 971, 563
    w_big, h_big = 0, 0
    # ----------------------------------杨----------------------------------------
    # classname = {0: '01-Nimitz Aircraft Carrier', 1: '06-Barracks Ship', 2: '05-Container Ship',
    #                     3: '06-Fishing Vessel', 4: '06-Henry J. Kaiser-class replenishment oiler',
    #                     5: '06-Other Warship', 6: '06-Yacht', 7: '06-Freedom-class littoral combat ship',
    #                     8: '02-Arleigh Burke-class Destroyer', 9: '06-Lewis and Clark-class dry cargo ship',
    #                     10: '06-Towing vessel', 11: '06-unknown', 12: '06-Powhatan-class tugboat',
    #                     13: '06-Barge', 14: '02-055-destroyer', 15: '02-052D-destroyer', 16: '06-USNS Bob Hope',
    #                     17: '06-USNS Montford Point', 18: '06-Bunker', 19: '06-Ticonderoga-class cruiser',
    #                     20: '03-Oliver Hazard Perry-class frigate',
    #                     21: '06-Sacramento-class fast combat support ship', 22: '06-Submarine',
    #                     23: '06-Emory S. Land-class submarine tender', 24: '06-Hatakaze-class destroyer',
    #                     25: '02-Murasame-class destroyer', 26: '06-Whidbey Island-class dock landing ship',
    #                     27: '06-Hiuchi-class auxiliary multi-purpose support ship', 28: '06-USNS Spearhead',
    #                     29: '02-Hyuga-class helicopter destroyer', 30: '02-Akizuki-class destroyer',
    #                     31: '05-Bulk carrier', 32: '02-Kongo-class destroyer', 33: '06-Northampton-class tug',
    #                     34: '05-Sand Carrier', 35: '06-Iowa-class battle ship',
    #                     36: '06-Independence-class littoral combat ship',
    #                     37: '06-Tarawa-class amphibious assault ship', 38: '04-Cyclone-class patrol ship',
    #                     39: '06-Wasp-class amphibious assault ship', 40: '06-074-landing ship',
    #                     41: '06-056-corvette', 42: '06-721-transport boat', 43: '06-037II-missile boat',
    #                     44: '06-Traffic boat', 45: '06-037-submarine chaser', 46: '06-unknown auxiliary ship',
    #                     47: '06-072III-landing ship', 48: '06-636-hydrographic survey ship',
    #                     49: '06-272-icebreaker', 50: '06-529-Minesweeper', 51: '03-053H2G-frigate',
    #                     52: '06-909A-experimental ship', 53: '06-909-experimental ship',
    #                     54: '06-037-hospital ship', 55: '06-Tuzhong Class Salvage Tug',
    #                     56: '02-022-missile boat', 57: '02-051-destroyer', 58: '03-054A-frigate',
    #                     59: '06-082II-Minesweeper', 60: '03-053H1G-frigate', 61: '06-Tank ship',
    #                     62: '02-Hatsuyuki-class destroyer', 63: '06-Sugashima-class minesweepers',
    #                     64: '06-YG-203 class yard gasoline oiler',
    #                     65: '06-Hayabusa-class guided-missile patrol boats', 66: '06-JS Chihaya',
    #                     67: '06-Kurobe-class training support ship', 68: '03-Abukuma-class destroyer escort',
    #                     69: '06-Uwajima-class minesweepers', 70: '06-Osumi-class landing ship',
    #                     71: '06-Hibiki-class ocean surveillance ships',
    #                     72: '06-JMSDF LCU-2001 class utility landing crafts', 73: '02-Asagiri-class Destroyer',
    #                     74: '06-Uraga-class Minesweeper Tender', 75: '06-Tenryu-class training support ship',
    #                     76: '06-YW-17 Class Yard Water', 77: '02-Izumo-class helicopter destroyer',
    #                     78: '06-Towada-class replenishment oilers', 79: '02-Takanami-class destroyer',
    #                     80: '06-YO-25 class yard oiler', 81: '06-891A-training ship', 82: '03-053H3-frigate',
    #                     83: '06-922A-Salvage lifeboat', 84: '06-680-training ship', 85: '06-679-training ship',
    #                     86: '06-072A-landing ship', 87: '06-072II-landing ship',
    #                     88: '06-Mashu-class replenishment oilers', 89: '06-903A-replenishment ship',
    #                     90: '06-815A-spy ship', 91: '06-901-fast combat support ship',
    #                     92: '06-Xu Xiake barracks ship', 93: '06-San Antonio-class amphibious transport dock',
    #                     94: '06-908-replenishment ship', 95: '02-052B-destroyer',
    #                     96: '06-904-general stores issue ship', 97: '02-051B-destroyer',
    #                     98: '06-925-Ocean salvage lifeboat', 99: '06-904B-general stores issue ship',
    #                     100: '06-625C-Oceanographic Survey Ship', 101: '06-071-amphibious transport dock',
    #                     102: '02-052C-destroyer', 103: '06-635-hydrographic Survey Ship',
    #                     104: '06-926-submarine support ship', 105: '06-917-lifeboat',
    #                     106: '06-Mercy-class hospital ship',
    #                     107: '06-Lewis B. Puller-class expeditionary mobile base ship',
    #                     108: '06-Avenger-class mine countermeasures ship', 109: '02-Zumwalt-class destroyer',
    #                     110: '06-920-hospital ship', 111: '02-052-destroyer', 112: '06-054-frigate',
    #                     113: '02-051C-destroyer', 114: '06-903-replenishment ship', 115: '06-073-landing ship',
    #                     116: '06-074A-landing ship', 117: '06-North Transfer 990',
    #                     118: '01-001-aircraft carrier', 119: '06-905-replenishment ship',
    #                     120: '06-Hatsushima-class minesweeper', 121: '01-Forrestal-class Aircraft Carrier',
    #                     122: '01-Kitty Hawk class aircraft carrier', 123: '06-Blue Ridge class command ship',
    #                     124: '06-081-Minesweeper', 125: '06-648-submarine repair ship',
    #                     126: '06-639A-Hydroacoustic measuring ship', 127: '06-JS Kurihama', 128: '06-JS Suma',
    #                     129: '06-Futami-class hydro-graphic survey ships', 130: '06-Yaeyama-class minesweeper',
    #                     131: '06-815-spy ship', 132: '02-Sovremenny-class destroyer'}
    # ----------------------------------杨----------------------------------------
    if not Use_Mask:
        # image = cv2.imread(r"G:\flask\images\b.tif")
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box["box"][:6]
            x_center_longitude, y_center_longitude = box["longitude"][:2]
            # 获取x的度分秒
            # parts_x = x_center_longitude.replace('°', "'").replace('"', '').split("'")
            # degrees_x = parts_x[0]
            # minutes_x = parts_x[1]
            # seconds_x = parts_x[2]
            # #获取y的度分秒
            # parts_y = y_center_longitude.replace('°', "'").replace('"', '').split("'")
            # degrees_y = parts_y[0]
            # minutes_y = parts_y[1]
            # seconds_y = parts_y[2]

            x1 += w_big
            x2 += w_big
            y1 += h_big
            y2 += h_big
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            clss = classname[cls.item()]
            colors = color[int(cls.item())]
            cv2.rectangle(image, (x1, y1), (x2, y2), colors, thickness=3)
            # label = f"Class: {clss}, Conf: {conf:.2f}"
            label = f"{clss},{conf:.2f},{x_center_longitude},{y_center_longitude}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
    else:
        height, width, _ = image.shape
        # 这里将超级大图放入
        # image = cv2.imread(r"G:\flask\images\b.tif")
        # 这里假设大图是RGB格式的
        mask_array = np.zeros_like(image)
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box["box"][:6]
            # 裁剪的机场大图加上左上角的坐标
            x1 += w_big
            x2 += w_big
            y1 += h_big
            y2 += h_big
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            clss = classname[cls.item()]
            colors = color[int(cls.item())]
            # cv2.rectangle(image, (x1, y1), (x2, y2), colors, thickness=3)
            # label = f"Class: {clss}, Conf: {conf:.2f}"
            label = f"Class: {clss}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
            # 遍历masks列表，将每个掩码绘制到大图上
            mask, x1, y1 = box["mask"][:4]
            x1 += w_big
            y1 += h_big
            mask = mask.data.cpu().numpy()
            # mask在predict后设置的resize的大小等于resize，不跟之前crop的大小相同，所以要重新resize回去画mask
            mask = cv2.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)
            # print(mask)
            # 确定掩码在大图上的位置
            x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
            y2 = min(y2, height + h_big)
            x2 = min(x2, width + w_big)
            mask = mask[:y2 - y1, :x2 - x1]
            color_mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            color_mask[mask == 1] = colors  # 只在掩码为1的位置上色
            mask_array[y1:y2, x1:x2][mask == 1] = color_mask[mask == 1]
        # 将掩码叠加到大图上
        # 这里使用了加权和来保持背景的可见性，你可以调整权重以满足你的需求
        image = cv2.addWeighted(image, 1.0, mask_array, 0.5, 0)

    return image


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
            detection_info = data.detection_info[j]['Longitude']
            conf = round(data.boxes.conf[j].item(), 3)
            xywh = data.boxes.xywh[j].cpu().detach().numpy().tolist()
            corner = [
                [round(xywh[0], 3), round(xywh[1], 3)],  # Left-top
                [round(xywh[0] + xywh[2], 3), round(xywh[1], 3)],  # Right-top
                [round(xywh[0] + xywh[2], 3), round(xywh[1] + xywh[3], 3)],  # Right-bottom
                [round(xywh[0], 3), round(xywh[1] + xywh[3], 3)]  # Left-bottom
            ]
            # masks = data.masks.xy[j].tolist()
            x, y, width, height = xywh
            target_width = round(width, 3)
            target_height = round(height, 3)
            # Pixelnumber = len(masks)
            lslist = {
                "class": cls,
                "conf": conf,
                "corner": corner,
                "coordinate": detection_info,
                "target_width": target_width,
                "target_height": target_height,
                # "masks_number": Pixelnumber,
                # "masks": masks,
            }
            data_list.append(lslist)
        json_data = json.dumps(data_list, indent=4)
        with open(file_path, "w") as file:
            file.write(json_data)

        all_results.append({json_filename: data_list})

    return all_results


def infrence(img_path, weights_path, crop_size, overlap_size, Use_Mask):
    json_output_dir = os.path.basename(img_path).replace(".", "_") + '_json'  # json保存路径
    file_name_without_extension = os.path.splitext(os.path.basename(img_path))[0]
    svimg = file_name_without_extension + '-output.tif'
    postfix = 'json'  # 保存格式txt或json
    Use_Masks = Use_Mask
    # 判断采用mask还是水平框

    results, img, colorss, names, all_Port_results, weights_path = inference(weights_path, img_path, crop_size,
                                                                             overlap_size,
                                                                             savejson, json_output_dir, postfix,
                                                                             Use_Masks)
    if len(results) > 0:
        # merged_results = torch.stack(results)
        img_with_boxes = draw_boxes(img, results, colorss, names, weights_path, crop_size, Use_Mask)
        # # ----------画5等分经纬度线-----------
        # from image_transfer import draw_longitude
        # draw_longitude(svimg, img_path, img_with_boxes)
        # # ----------画5等分经纬度线-----------
        cv2.imwrite(svimg, img_with_boxes)
        img_str = image_to_base64(img_with_boxes)
        all_Port_results.append({"image": img_str})
        print(f'已将图片保存在{svimg}')
    else:
        print("没有检测到任何目标。")


def image_to_base64(img):
    # 将图像转换为JPEG格式
    _, buffer = cv2.imencode('.jpg', img)
    # 将图像数据转换为base64编码的字符串
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


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

    # weights_path = r'./weights/airports.pt'
    # img_path = r"./images/a.tif"

    # 解析接口参数
    # data = request.get_data()
    # data = json.loads(data)

    # 权重路径
    weights_dict = {
        '1': r'weights\airports.pt',
        '2': r'weights\ship50.pt',
        '3': r'weights\airplane.pt',
        '4': r'weights\oilTank.pt',
        '5': r'weights\harbor.pt',
        '6': r'weights\bridge.pt',
        '7': r'weights\oil.pt',
        '8': r'weights\gangkou1474.pt',
    }

    # 根据参数选着权重
    weights_type = request.args.get("weights_type")
    weights_path = weights_dict[str(weights_type)]  # 权重

    # img_dict = {
    #     '1': r'images\a.tif',
    #     '2': r'images\b.tif',
    #     '3': r'港口部件_oil_1024x1024.tif',
    #     '4': r'crop1.png',
    #     '5': r'crop6-(13681, 8625, 17265, 11409).png',
    #     '6': r'crop1-(33, 6314, 6017, 8922).png',
    #     '7': r'images\SpaceNet_3.jpg',
    #     '8': r'images\SpaceNet_6.jpg',
    #     '9': r'zy.tif',
    #     '11': r'ZY_QQ/b_cut.png',
    #     '12': r'T1.jpg',
    #     '13': r'T2.jpg',
    #     '14': r'T3.jpg',
    #     '15': r'T4.jpg',
    #     '16': r'T5.jpg',
    #     '17': r'S1.jpg',
    #     '18': r'TW_TY.jpg',
    # }

    # ZFJ start
    pic_name = request.args.get("pic_name")
    num_rows = request.args.get("num_rows")
    pil_img_dict = scene_classification(pic_name, int(num_rows))
    # img_dict: {"Airport":[pic1_path, pic2_path,...], "Port":[pic1_path, pic2_path,...]}
    # zfj end
    img_path_Airport = pil_img_dict['Airport']
    img_path_Port = pil_img_dict['Port']

    # 图片路径
    # img_type = request.args.get("img_type")

    overlap_size = 100  # 重叠区域的大小
    crop_size = (640, 640)  # 裁剪大小
    #  检测飞机
    if weights_path == 'weights\\airplane.pt':
        Use_Mask = False
        if img_path_Airport:
            all_result = []
            for img_path in img_path_Airport:
                infrence(img_path, weights_path, crop_size, overlap_size, Use_Mask)
    # 检测机场
    if weights_path == 'weights\\airports.pt':
        Use_Mask = True
        if img_path_Airport:
            all_result = []
            for img_path in img_path_Airport:
                infrence(img_path, weights_path, crop_size, overlap_size, Use_Mask)
    # 检测 舰船
    if weights_path == 'weights\\ship50.pt':
        Use_Mask = True
        if img_path_Port:
            all_result = []
            for img_path in img_path_Port:
                infrence(img_path, weights_path, crop_size, overlap_size, Use_Mask)
    else:
        Use_Mask = False

    return jsonify(all_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
