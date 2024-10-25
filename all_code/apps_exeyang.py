import base64
import logging
import cv2
import torch
from flask import Flask
from flask_cors import CORS
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
import sys
from main_run import scene_classification

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)

from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)


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


def inference(weights_path, img_path, crop_size, overlap_size, savejson_func, json_output_dir, postfix, pil_img_dict):
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
    # 切割图像并进行预测
    results = []
    all_results = []
    tile_id = 0
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
            result = model.predict(cropped_image, imgsz=640)
            names = result[0].names
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

            # 如果结果中有检测框
            if len(result[0].boxes.data) > 0:
                # 如果权重路径是'weights\\oil.pt'，'weights\\airplane.pt'或'weights\\harbor.pt'
                if weights_path == 'weights\\oil.pt' or weights_path == 'weights\\airplane.pt' or weights_path == 'weights\\harbor.pt':
                    # 遍历检测框
                    for det in result[0].boxes.data:
                        # 克隆张量，避免原地修改
                        det = det.clone()
                        x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
                        # 转换坐标（使其相对于原图）
                        det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box
                        # 添加到结果列表
                        combined_results.append({
                            'box': det,  # 检测框
                        })
                        # 转换为numpy数组并添加到检测信息列表
                        det = det.cpu().detach().numpy().tolist()
                        detection_info.append({"bbox": [round(num, 3) for num in det[:4]]})
                    # 更新结果的检测信息
                    result[0].detection_info = detection_info
                else:
                    # 遍历检测框和掩码
                    for det, msk in zip(result[0].boxes.data, result[0].masks.data):
                        # 克隆张量，避免原地修改
                        det = det.clone()
                        x1_box, y1_box, x2_box, y2_box, conf, cls = det[:6]
                        # 转换坐标（使其相对于原图）
                        det[0], det[1], det[2], det[3] = x + x1_box, y + y1_box, x + x2_box, y + y2_box
                        # 添加到结果列表
                        combined_results.append({
                            'box': det,  # 检测框
                            'mask': [msk, x, y],  # 掩码
                        })
                        # 转换为numpy数组并添加到检测信息列表
                        det = det.cpu().detach().numpy().tolist()
                        detection_info.append({"bbox": [round(num, 3) for num in det[:4]]})
                    # 更新结果的检测信息
                    result[0].detection_info = detection_info
            # 如果savejson_func不为空
            if savejson_func is not None:
                # 创建json文件名
                json_filename = f"image_{tile_id}.{postfix}"
                # 创建json文件路径
                json_file_path = os.path.join(json_output_dir, json_filename)
                # 保存json文件
                all_results = savejson_func(result, json_file_path, names, all_results, json_filename)
                # tile_id增加1
                tile_id += 1
    # 提取坐标并转换为张量
    coords = torch.stack([item['box'][:4] for item in combined_results])
    # 提取分数
    scores = torch.stack([item['box'][4] for item in combined_results])
    # 进行非极大值抑制
    keep = nms(coords, scores, iou_threshold=0.5)
    # 获取非极大值抑制后的结果
    nms_results = [combined_results[i] for i in keep]
    # 将nms_results添加到results中
    results.extend(nms_results)
    return results, img, colorss, names, all_results, weights_path


# 定义一个函数来根据类别选择颜色
def get_color_by_class(cls):
    # 这里你可以定义一个颜色映射
    color_map = {
        0: (255, 0, 0),  # 类别0的颜色（例如，红色）
        1: (0, 255, 0),  # 类别1的颜色（例如，绿色）
        # ...为其他类别定义颜色
    }
    return color_map.get(cls, (255, 255, 255))  # 默认颜色为白色


def draw_boxes(image, boxes, color, classname, weights_path, crop_size,Use_Masks):
    # 如果权重路径是'weights\\oil.pt'，'weights\\airplane.pt'或'weights\\harbor.pt'
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
            label = f"Class: {clss}, Conf: {conf:.2f}"
            # 在图像上添加标签
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)
    # 如果权重路径不是上述三个之一
    else:
        # 获取图像的高度和宽度
        height, width, _ = image.shape
        # 创建一个和大图同样大小的空白图像，用于绘制掩码
        mask_array = np.zeros_like(image)
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
            label = f"Class: {clss}, Conf: {conf:.2f}"
            # 在图像上添加标签
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colors, thickness=2)

            # 获取掩码和其在大图上的位置
            mask, x1, y1 = box["mask"][:4]
            # 将掩码转换为numpy数组
            mask = mask.data.cpu().numpy()
            # 将掩码调整为和裁剪图像同样的大小
            mask = cv2.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)
            # 确定掩码在大图上的位置
            x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
            # 确保掩码不超过大图的边界
            y2 = min(y2, height)
            x2 = min(x2, width)
            # 调整掩码的大小以适应大图
            mask = mask[:y2 - y1, :x2 - x1]
            # 创建一个和掩码同样大小的空白图像，用于上色
            color_mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            # 只在掩码为1的位置上色
            color_mask[mask == 1] = colors
            # 将上色后的掩码添加到大图的掩码图像上
            mask_array[y1:y2, x1:x2][mask == 1] = color_mask[mask == 1]
        # 将掩码图像叠加到大图上
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
            detection_info = data.detection_info[j]['bbox']
            conf = round(data.boxes.conf[j].item(), 3)
            xywh = data.boxes.xywh[j].cpu().detach().numpy().tolist()
            corner = [
                [round(xywh[0], 3), round(xywh[1], 3)],  # 左上角
                [round(xywh[0] + xywh[2], 3), round(xywh[1], 3)],  # 右上角
                [round(xywh[0] + xywh[2], 3), round(xywh[1] + xywh[3], 3)],  # 右下角
                [round(xywh[0], 3), round(xywh[1] + xywh[3], 3)]  # 左下角
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
        img_with_boxes = draw_boxes(img, results, colorss, names, weights_path, crop_size, Use_Masks)
        # # ----------画5等分经纬度线-----------
        # from image_transfer import draw_longitude
        # draw_longitude(svimg, img_path, img_with_boxes)
        # # ----------画5等分经纬度线-----------
        cv2.imwrite(svimg, img_with_boxes)
        img_str = image_to_base64(img_with_boxes)
        # all_Port_results.append({"image": img_str})
        print(f'已将图片保存在{svimg}')
    else:
        print("没有检测到任何目标。")
    return all_Port_results

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return image_data


def convert_image_data(image_data):
    # 将图像数据转换为numpy数组
    np_array = np.frombuffer(image_data, np.uint8)
    # 使用OpenCV的imdecode函数将numpy数组转换为OpenCV的图像格式
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


# def display_image(image):
#     cv2.namedWindow("Image", cv2.WINDOW_FREERATIO)
#     # 使用OpenCV的imshow函数显示图像
#     cv2.imshow("Image", image)
#     # 等待按下任意键后关闭窗口
#     cv2.waitKey(0)
#     # 关闭窗口
#     cv2.destroyAllWindows()

import requests

def Rio_report():
    base_url = 'http://127.0.0.1:5000/classify'
    # 定义参数
    params = {
        'image_path':'',
        'pic_name': "zy",
        'num_rows': 29,
    }

    # 发送 GET 请求
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data



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
        '3': r'weights\oil.pt',
        '4': r'weights\airplane.pt',
        '5': r'weights\harbor.pt',
        '6': r'F:\airport\exp\weights\best.pt'
    }

    # data = Rio_report()
    # classify_result_json_path = data
    classify_result_json_paths = 'zy.json'
    with open(classify_result_json_paths, 'r') as json_file:
        # 使用json.load()加载JSON数据
        all_data = json.load(json_file)

    # 根据参数选着权重
    weights_type = request.args.get("weights_type")
    weights_path = weights_dict[str(weights_type)]  # 权重


    img_path_Airport = []
    img_path_Port = []

    Airport_Pic = convert_image_data(base64_to_image(all_data["Ori_Pic"][0]))  # 原图
    if all_data['name'] =='qq':
        Airport_Pos = all_data["Airport_ROI_Pos"] # 裁剪点位
        Airport_ROI_Pic = all_data["Airport_ROI_Pic"][0]  # 裁剪图像
    else:
        Airport_Pos = all_data["Airport_ROI_Pos"] # 裁剪点位
        Airport_ROI_Pic = all_data["Airport_ROI_Pic"] # 裁剪图像
        Port_Pos = all_data["Port_ROI_Pos"] # 裁剪点位
        Port_ROI_Pic = all_data["Port_ROI_Pic"] # 裁剪图像



    # 图片路径
    # img_type = ("img_type")

    overlap_size = 100  # 重叠区域的大小
    crop_size = (640, 640)  # 裁剪大小
    #  检测飞机
    all_result = []
    # Use_Mask = True
    # out = infrence(img_path_Port, weights_path, crop_size, overlap_size, Use_Mask)

    if weights_path == 'weights\\airplane.pt':
        Use_Mask = False
        if Airport_ROI_Pic:
            for img_path in Airport_ROI_Pic:
                out = infrence(img_path, weights_path, crop_size, overlap_size, Use_Mask)
                all_result.append(out)
    # 检测机场
    elif weights_path == 'weights\\airport.pt':
        Use_Mask = True
        if Airport_ROI_Pic:
            for img_path in Airport_ROI_Pic:
                out = infrence(img_path, weights_path, crop_size, overlap_size, Use_Mask)
                all_result.append(out)
    # 检测 舰船
    elif weights_path == 'weights\\ship50.pt':
        Use_Mask = True
        if Port_ROI_Pic:
            for img_path in Port_ROI_Pic:
                out = infrence(img_path, weights_path, crop_size, overlap_size, Use_Mask)
                all_result.append(out)
    else:
        Use_Mask = False
        for img_path in Port_ROI_Pic:
            out = infrence(img_path, weights_path, crop_size, overlap_size, Use_Mask)
            all_result.append(out)

    return jsonify(all_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)
