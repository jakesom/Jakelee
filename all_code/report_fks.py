import json
import requests
import base64
import numpy as np
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return image_data


def convert_image_data(image_data):
    # 将图像数据转换为numpy数组
    np_array = np.frombuffer(image_data, np.uint8)
    # 使用OpenCV的imdecode函数将numpy数组转换为OpenCV的图像格式
    image = cv2.imdecode(np_array, cv2.IMREADi_COLOR)
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
            '6': r'G:\flask_all\flask\best.pt'
        }
    '''
    # 发送 GET 请求
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data



def classify_Rio_reports():
    base_url = 'http://127.0.0.1:5006/classify'
    # 定义参数
    params = {
        'ori_path': 'zy.tif',
        'pic_name': "zy",
        'num_rows': 29,
    }
    # 发送 GET 请求
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        data = response.json()
        return data





if __name__ == '__main__':

    import time
    begin_time = time.time()
    # data = classify_Rio_reports()
    # print(data)
    # ori_path = data["json_file_path"]
    ori_path = "result_json/ff.json"
    end_time = time.time()
    print(f"场景运行时间为{end_time-begin_time}")
    # ori_path = r"result_json/zy.json"
    begin_time = time.time()
    end_data = detect_Rio_report(ori_path)
    end_time = time.time()
    print(f"检测运行时间为{end_time - begin_time}")
