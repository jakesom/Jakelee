import json
import os

import requests

import base64
import numpy as np
import cv2


def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return image_data


def convert_image_data(image_data):
    # 将图像数据转换为numpy数组
    np_array = np.frombuffer(image_data, np.uint8)
    # 使用OpenCV的imdecode函数将numpy数组转换为OpenCV的图像格式
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


def display_image(image):
    cv2.namedWindow("Image", cv2.WINDOW_FREERATIO)
    # 使用OpenCV的imshow函数显示图像
    cv2.imshow("Image", image)
    # 等待按下任意键后关闭窗口
    cv2.waitKey(0)
    # 关闭窗口
    cv2.destroyAllWindows()



def detect_Rio_report(ori_path):
    base_url = 'http://127.0.0.1:5000/start'
    # 定义参数

    ori_paths = ori_path
    params = {
        'ori_path': ori_paths,
        'weights_type': 3,
    }
    '''
        weights_dict = {
            '1': r'weights\airport.pt',
            '2': r'weights\ship50.pt',
            '3': r'weights\airplane.pt',
            '4': r'weights\oil.pt',
            '5': r'weights\harbor.pt',
            '6': r'weights\airport2.pt',
        }
    '''
    # 发送 GET 请求
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("请求错误")
        return response.status_code



def classify_Rio_reports():
    pic_name = 'qq'
    # postfix = 'json'
    # json_filename = f"{pic_name}.{postfix}"
    # json_output_dir = './result_json/'
    # json_file_path = os.path.join(json_output_dir, json_filename)
    # if os.path.exists(json_file_path):
    #     return json_file_path
    # else:
    base_url = 'http://127.0.0.1:5006/classify'
    # 定义参数
    params = {
        'ori_path': 'qq.tif',
        'pic_name': pic_name,
        'num_rows': 32,
    }
    # 发送 GET 请求
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        data = response.json()
        return data




if __name__ == '__main__':

    detect_Rio_report()

    # import time
    # begin_time = time.time()
    #
    # data = classify_Rio_reports()
    # print(data)
    # # ori_path = data
    # #
    # # # ori_path = "result_json/zy.json"
    # # end_time = time.time()
    # # print(f"场景运行时间为{end_time-begin_time}")
    # # # ori_path = r"result_json/zy.json"
    # # begin_time = time.time()
    # # end_data = detect_Rio_report(ori_path)
    # # end_time = time.time()
    # # print(f"检测运行时间为{end_time - begin_time}")
