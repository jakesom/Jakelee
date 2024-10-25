import shutil
from collections import Counter
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from data import MODEL


# 大图裁剪成小图
def large_patches(pic_name=None, current_dir=None, large_image=None, output_patches_folder=None, patch_size=None):
    # 设置小图的大小和重叠率
    small_width = patch_size  # 小图的宽度
    small_height = patch_size  # 小图的高度
    overlap = 0  # 重叠像素个数
    start_row = -1
    start_col = -1

    # 获取大图的宽度和高度
    large_width, large_height = large_image.size

    os.makedirs(output_patches_folder, exist_ok=True)

    if os.path.exists(output_patches_folder) is True:
        shutil.rmtree(output_patches_folder)
        os.makedirs(output_patches_folder, exist_ok=True)

    # 循环切割大图并保存小图
    for y in range(0, large_height, small_height - overlap):
        start_row += 1
        for x in range(0, large_width, small_width - overlap):
            start_col += 1
            # 切割大图
            box = (x, y, x + small_width, y + small_height)
            small_image = large_image.crop(box)
            filename = os.path.join(output_patches_folder, f"{start_row}_{start_col}+Category.tif")
            small_image.save(filename)
        start_col = -1


def cat_imgs_inner_and_surrounding(output_folder=None, output_folder_surrounding=None, input_folder=None):
    # 检查文件夹是否存在
    if not os.path.exists(output_folder):
        # 如果文件夹不存在，则创建它
        os.makedirs(output_folder)

    # 检查文件夹是否存在
    if not os.path.exists(output_folder_surrounding):
        # 如果文件夹不存在，则创建它
        os.makedirs(output_folder_surrounding)

    # 确定小图像的行数和列数
    num_rows = 0
    num_cols = 0

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            parts = filename.split('_')
            r, c = int(parts[0]), int(parts[1].split("+")[0])
            num_rows = max(num_rows, r + 1)
            num_cols = max(num_cols, c + 1)

    stride = 1

    def find_jpg_files_with_string(folder_path, target_string):
        jpg_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.tif') and file.startswith(target_string):
                    jpg_files.append(os.path.join(root, file))
        return jpg_files

    def read_single_file(input_folder, start_name):
        filenames = find_jpg_files_with_string(input_folder, start_name)
        assert len(filenames) == 1
        filename = filenames[0]
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)
        return img, filename

    top_left = []
    top_right = []
    bottom_left = []
    bottom_right = []
    top_line = []
    bottom_line = []
    left_line = []
    right_line = []

    for r in range(0, num_rows, stride):
        for c in range(0, num_cols, stride):
            if r == 0 and c == 0:
                top_left.append([r, c])
            elif r == 0 and c == num_cols - 1:
                top_right.append([r, c])
            elif r == num_rows - 1 and c == 0:
                bottom_left.append([r, c])
            elif r == num_rows - 1 and c == num_cols - 1:
                bottom_right.append([r, c])
            elif r == 0 and c != 0 and c != num_cols - 1:
                top_line.append([r, c])
            elif r == num_rows - 1 and c != 0 and c != num_cols - 1:
                bottom_line.append([r, c])
            elif r != 0 and r != num_rows - 1 and c == 0:
                left_line.append([r, c])
            elif r != 0 and r != num_rows - 1 and c == num_cols - 1:
                right_line.append([r, c])

    for top_left_pos in top_left:
        r = top_left_pos[0]
        c = top_left_pos[1]
        start_name = str(r) + "_" + str(c) + "+"
        center_img, filename = read_single_file(input_folder, start_name)
        img_left_right = center_img.transpose(Image.FLIP_LEFT_RIGHT)
        img_top_bottom = center_img.transpose(Image.FLIP_TOP_BOTTOM)
        img0 = center_img.rotate(180)
        img1 = img_top_bottom
        img3 = img_left_right
        img4 = center_img
        img5, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c) + "+")
        img6 = img5.transpose(Image.FLIP_LEFT_RIGHT)
        img7, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c + 1) + "+")
        img2 = img7.transpose(Image.FLIP_TOP_BOTTOM)
        img8, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c + 1) + "+")
        result_image = Image.new('RGB', (3 * 408, 3 * 408))
        result_image.paste(img0, (0 * 408, 0 * 408))
        result_image.paste(img1, (1 * 408, 0 * 408))
        result_image.paste(img6, (0 * 408, 2 * 408))
        result_image.paste(img3, (0 * 408, 1 * 408))
        result_image.paste(img4, (1 * 408, 1 * 408))
        result_image.paste(img5, (1 * 408, 2 * 408))
        result_image.paste(img2, (2 * 408, 0 * 408))
        result_image.paste(img7, (2 * 408, 1 * 408))
        result_image.paste(img8, (2 * 408, 2 * 408))
        center_catgory = filename.split(".")[0].split("+")[1]
        save_file_name = str(r) + "_" + str(c) + "+" + center_catgory + '.tif'
        output_path1 = os.path.join(output_folder_surrounding, save_file_name)
        result_image.save(output_path1)

    for top_right_pos in top_right:
        r = top_right_pos[0]
        c = top_right_pos[1]
        start_name = str(r) + "_" + str(c) + "+"
        center_img, filename = read_single_file(input_folder, start_name)
        img_left_right = center_img.transpose(Image.FLIP_LEFT_RIGHT)
        img_top_bottom = center_img.transpose(Image.FLIP_TOP_BOTTOM)
        img0 = center_img.rotate(180)
        img1 = img_top_bottom
        img2 = center_img.rotate(180)
        img3, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c - 1) + "+")
        img4 = center_img
        img5 = img_left_right
        img6, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c - 1) + "+")
        img7, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c) + "+")
        img8 = img7.transpose(Image.FLIP_LEFT_RIGHT)

        result_image = Image.new('RGB', (3 * 408, 3 * 408))
        result_image.paste(img0, (0 * 408, 0 * 408))
        result_image.paste(img1, (1 * 408, 0 * 408))
        result_image.paste(img2, (2 * 408, 0 * 408))
        result_image.paste(img3, (0 * 408, 1 * 408))
        result_image.paste(img4, (1 * 408, 1 * 408))
        result_image.paste(img5, (2 * 408, 1 * 408))
        result_image.paste(img6, (0 * 408, 2 * 408))
        result_image.paste(img7, (1 * 408, 2 * 408))
        result_image.paste(img8, (2 * 408, 2 * 408))

        center_catgory = filename.split(".")[0].split("+")[1]
        save_file_name = str(r) + "_" + str(c) + "+" + center_catgory + '.tif'
        save_file_name2 = str(r) + "_" + str(c) + "_Center" + "+" + center_catgory + '.tif'
        output_path1 = os.path.join(output_folder_surrounding, save_file_name)
        result_image.save(output_path1)

    for bottom_left_pos in bottom_left:
        r = bottom_left_pos[0]
        c = bottom_left_pos[1]
        start_name = str(r) + "_" + str(c) + "+"
        center_img, filename = read_single_file(input_folder, start_name)
        img_left_right = center_img.transpose(Image.FLIP_LEFT_RIGHT)
        img_top_bottom = center_img.transpose(Image.FLIP_TOP_BOTTOM)
        img6 = center_img.rotate(180)
        img7 = img_top_bottom
        img3 = img_left_right
        img4 = center_img
        img1, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c) + "+")
        img0 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c + 1) + "+")
        img5, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c + 1) + "+")
        img8 = img5.transpose(Image.FLIP_TOP_BOTTOM)
        result_image = Image.new('RGB', (3 * 408, 3 * 408))
        result_image.paste(img0, (0 * 408, 0 * 408))
        result_image.paste(img1, (1 * 408, 0 * 408))
        result_image.paste(img2, (2 * 408, 0 * 408))
        result_image.paste(img3, (0 * 408, 1 * 408))
        result_image.paste(img4, (1 * 408, 1 * 408))
        result_image.paste(img5, (2 * 408, 1 * 408))
        result_image.paste(img6, (0 * 408, 2 * 408))
        result_image.paste(img7, (1 * 408, 2 * 408))
        result_image.paste(img8, (2 * 408, 2 * 408))
        center_catgory = filename.split(".")[0].split("+")[1]
        save_file_name = str(r) + "_" + str(c) + "+" + center_catgory + '.tif'
        save_file_name2 = str(r) + "_" + str(c) + "_Center" + "+" + center_catgory + '.tif'
        output_path1 = os.path.join(output_folder_surrounding, save_file_name)
        result_image.save(output_path1)

    for bottom_right_pos in bottom_right:
        r = bottom_right_pos[0]
        c = bottom_right_pos[1]
        start_name = str(r) + "_" + str(c) + "+"
        center_img, filename = read_single_file(input_folder, start_name)
        img0, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c - 1) + "+")
        img1, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c) + "+")
        img2 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img3, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c - 1) + "+")
        img4, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c) + "+")
        img5 = img4.transpose(Image.FLIP_LEFT_RIGHT)
        img6 = img3.transpose(Image.FLIP_TOP_BOTTOM)
        img7 = img4.transpose(Image.FLIP_TOP_BOTTOM)
        img8 = center_img.rotate(180)
        result_image = Image.new('RGB', (3 * 408, 3 * 408))
        result_image.paste(img0, (0 * 408, 0 * 408))
        result_image.paste(img1, (1 * 408, 0 * 408))
        result_image.paste(img2, (2 * 408, 0 * 408))
        result_image.paste(img3, (0 * 408, 1 * 408))
        result_image.paste(img4, (1 * 408, 1 * 408))
        result_image.paste(img5, (2 * 408, 1 * 408))
        result_image.paste(img6, (0 * 408, 2 * 408))
        result_image.paste(img7, (1 * 408, 2 * 408))
        result_image.paste(img8, (2 * 408, 2 * 408))
        center_catgory = filename.split(".")[0].split("+")[1]
        save_file_name = str(r) + "_" + str(c) + "+" + center_catgory + '.tif'
        save_file_name2 = str(r) + "_" + str(c) + "_Center" + "+" + center_catgory + '.tif'
        output_path1 = os.path.join(output_folder_surrounding, save_file_name)
        result_image.save(output_path1)

    for top_line_pos in top_line:
        r = top_line_pos[0]
        c = top_line_pos[1]
        start_name = str(r) + "_" + str(c) + "+"
        filenames = find_jpg_files_with_string(input_folder, start_name)
        assert len(filenames) == 1
        filename = filenames[0]
        img3, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c - 1) + "+")
        img4, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c) + "+")
        img5, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c + 1) + "+")
        img6, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c - 1) + "+")
        img7, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c) + "+")
        img8, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c + 1) + "+")
        img0 = img3.transpose(Image.FLIP_TOP_BOTTOM)
        img1 = img4.transpose(Image.FLIP_TOP_BOTTOM)
        img2 = img5.transpose(Image.FLIP_TOP_BOTTOM)

        result_image = Image.new('RGB', (3 * 408, 3 * 408))
        result_image.paste(img0, (0 * 408, 0 * 408))
        result_image.paste(img1, (1 * 408, 0 * 408))
        result_image.paste(img2, (2 * 408, 0 * 408))
        result_image.paste(img3, (0 * 408, 1 * 408))
        result_image.paste(img4, (1 * 408, 1 * 408))
        result_image.paste(img5, (2 * 408, 1 * 408))
        result_image.paste(img6, (0 * 408, 2 * 408))
        result_image.paste(img7, (1 * 408, 2 * 408))
        result_image.paste(img8, (2 * 408, 2 * 408))

        center_catgory = filename.split(".")[0].split("+")[1]
        save_file_name = str(r) + "_" + str(c) + "+" + center_catgory + '.tif'
        save_file_name2 = str(r) + "_" + str(c) + "_Center" + "+" + center_catgory + '.tif'
        output_path1 = os.path.join(output_folder_surrounding, save_file_name)
        result_image.save(output_path1)

    for bottom_line_pos in bottom_line:
        r = bottom_line_pos[0]
        c = bottom_line_pos[1]
        start_name = str(r) + "_" + str(c) + "+"
        filenames = find_jpg_files_with_string(input_folder, start_name)
        assert len(filenames) == 1
        filename = filenames[0]
        img0, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c - 1) + "+")
        img1, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c) + "+")
        img2, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c + 1) + "+")
        img3, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c - 1) + "+")
        img4, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c) + "+")
        img5, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c + 1) + "+")
        img6 = img3.transpose(Image.FLIP_TOP_BOTTOM)
        img7 = img4.transpose(Image.FLIP_TOP_BOTTOM)
        img8 = img5.transpose(Image.FLIP_TOP_BOTTOM)

        result_image = Image.new('RGB', (3 * 408, 3 * 408))
        result_image.paste(img0, (0 * 408, 0 * 408))
        result_image.paste(img1, (1 * 408, 0 * 408))
        result_image.paste(img2, (2 * 408, 0 * 408))
        result_image.paste(img3, (0 * 408, 1 * 408))
        result_image.paste(img4, (1 * 408, 1 * 408))
        result_image.paste(img5, (2 * 408, 1 * 408))
        result_image.paste(img6, (0 * 408, 2 * 408))
        result_image.paste(img7, (1 * 408, 2 * 408))
        result_image.paste(img8, (2 * 408, 2 * 408))

        center_catgory = filename.split(".")[0].split("+")[1]
        save_file_name = str(r) + "_" + str(c) + "+" + center_catgory + '.tif'
        save_file_name2 = str(r) + "_" + str(c) + "_Center" + "+" + center_catgory + '.tif'
        output_path1 = os.path.join(output_folder_surrounding, save_file_name)
        result_image.save(output_path1)

    for left_line_pos in left_line:
        r = left_line_pos[0]
        c = left_line_pos[1]
        start_name = str(r) + "_" + str(c) + "+"
        filenames = find_jpg_files_with_string(input_folder, start_name)
        assert len(filenames) == 1
        filename = filenames[0]
        img1, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c) + "+")
        img2, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c + 1) + "+")
        img4, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c) + "+")
        img5, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c + 1) + "+")
        img7, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c) + "+")
        img8, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c + 1) + "+")
        img0 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img3 = img4.transpose(Image.FLIP_LEFT_RIGHT)
        img6 = img7.transpose(Image.FLIP_LEFT_RIGHT)

        result_image = Image.new('RGB', (3 * 408, 3 * 408))
        result_image.paste(img0, (0 * 408, 0 * 408))
        result_image.paste(img1, (1 * 408, 0 * 408))
        result_image.paste(img2, (2 * 408, 0 * 408))
        result_image.paste(img3, (0 * 408, 1 * 408))
        result_image.paste(img4, (1 * 408, 1 * 408))
        result_image.paste(img5, (2 * 408, 1 * 408))
        result_image.paste(img6, (0 * 408, 2 * 408))
        result_image.paste(img7, (1 * 408, 2 * 408))
        result_image.paste(img8, (2 * 408, 2 * 408))

        center_catgory = filename.split(".")[0].split("+")[1]
        save_file_name = str(r) + "_" + str(c) + "+" + center_catgory + '.tif'
        save_file_name2 = str(r) + "_" + str(c) + "_Center" + "+" + center_catgory + '.tif'
        output_path1 = os.path.join(output_folder_surrounding, save_file_name)
        output_path2 = os.path.join(output_folder_surrounding, save_file_name2)
        result_image.save(output_path1)

    for right_line_pos in right_line:
        r = right_line_pos[0]
        c = right_line_pos[1]
        start_name = str(r) + "_" + str(c) + "+"
        filenames = find_jpg_files_with_string(input_folder, start_name)
        assert len(filenames) == 1
        filename = filenames[0]
        img0, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c - 1) + "+")
        img1, _ = read_single_file(input_folder, start_name=str(r - 1) + "_" + str(c) + "+")
        img3, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c - 1) + "+")
        img4, _ = read_single_file(input_folder, start_name=str(r) + "_" + str(c) + "+")
        img6, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c - 1) + "+")
        img7, _ = read_single_file(input_folder, start_name=str(r + 1) + "_" + str(c) + "+")
        img2 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img5 = img4.transpose(Image.FLIP_LEFT_RIGHT)
        img8 = img7.transpose(Image.FLIP_LEFT_RIGHT)

        result_image = Image.new('RGB', (3 * 408, 3 * 408))
        result_image.paste(img0, (0 * 408, 0 * 408))
        result_image.paste(img1, (1 * 408, 0 * 408))
        result_image.paste(img2, (2 * 408, 0 * 408))
        result_image.paste(img3, (0 * 408, 1 * 408))
        result_image.paste(img4, (1 * 408, 1 * 408))
        result_image.paste(img5, (2 * 408, 1 * 408))
        result_image.paste(img6, (0 * 408, 2 * 408))
        result_image.paste(img7, (1 * 408, 2 * 408))
        result_image.paste(img8, (2 * 408, 2 * 408))

        center_catgory = filename.split(".")[0].split("+")[1]
        save_file_name = str(r) + "_" + str(c) + "+" + center_catgory + '.tif'
        save_file_name2 = str(r) + "_" + str(c) + "_Center" + "+" + center_catgory + '.tif'
        output_path1 = os.path.join(output_folder_surrounding, save_file_name)
        result_image.save(output_path1)

    # 逐行逐列拼接图像
    for r in range(0, num_rows - 2, stride):
        for c in range(0, num_cols - 2, stride):
            result_image = Image.new('RGB', (1224, 1224))
            start_name = str(r) + "_" + str(c) + "+"
            filenames = find_jpg_files_with_string(input_folder, start_name)
            assert len(filenames) == 1
            cat_file_list = []
            # 创建一个新的图像对象，用于拼接
            result_image = Image.new('RGB', (3 * 408, 3 * 408))
            imgs_3x3 = []
            center_catgory = None
            center_img = None
            count = 0
            for i in range(r, r + 3):
                for j in range(c, c + 3):
                    count += 1
                    cat_file_list.append((i, j))
                    start_name = str(i) + "_" + str(j) + "+"
                    filenames = find_jpg_files_with_string(input_folder, start_name)
                    assert len(filenames) == 1
                    filename = filenames[0]
                    image_path = os.path.join(input_folder, filename)
                    img = Image.open(image_path)
                    if count == 5:
                        center_catgory = filename.split(".")[0].split("+")[1]
                        center_img = img
                    imgs_3x3.append(img)
            # 按行列将图像拼接到九宫格中
            for i in range(3):
                for j in range(3):
                    img = imgs_3x3[i * 3 + j]
                    result_image.paste(img, (j * 408, i * 408))

            save_file_name = str(r + 1) + "_" + str(c + 1) + "+" + center_catgory + '.tif'
            save_file_name2 = str(r + 1) + "_" + str(c + 1) + "_Center" + "+" + center_catgory + '.tif'
            output_path1 = os.path.join(output_folder, save_file_name)
            result_image.save(output_path1)


def run_model(pic_name=None, pred_label_save_txt=None, imgs_root=None, json_path=None, weights_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load image
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".tif")]

    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = MODEL(num_classes=7, vit_pretrained_weights=None,
                  mae_pretrained_weights=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  embed_dim=1024, use_attention=True, use_center_mask=True).to(device)

    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    msg = model.load_state_dict(torch.load(weights_path, map_location=device))
    print(msg)

    # prediction
    model.eval()
    batch_size = 1  # 必须指定为1
    with torch.no_grad():
        with open(pred_label_save_txt, 'w', encoding='utf-8') as file:
            for ids in range(0, len(img_path_list) // batch_size):
                img_list = []
                img_list_center = []
                img_list_path = []
                for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                    img_list_path.append(img_path)
                    assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                    img = Image.open(img_path)
                    center_img = img.crop((408, 408, 816, 816))
                    # plt.imshow(img)
                    # plt.imshow(center_img)
                    img = data_transform(img)
                    center_img = data_transform(center_img)
                    img_list.append(img)
                    img_list_center.append(center_img)

                whole_images = torch.stack(img_list, dim=0)
                normal_images = torch.stack(img_list_center, dim=0)
                output, _ = model(normal_images.to(device), whole_images.to(device), mask_ratio=0.08)
                pred_classes = torch.max(output, dim=1)[1]
                pred_classes = pred_classes.cpu().numpy()
                for index, img_path in enumerate(img_list_path):
                    msg = img_path.replace("\\", "/").split("/")[-1] + " --> " + class_indict[str(pred_classes[index])]
                    file.write(msg + '\n')  # 在每行的末尾添加换行符
    file.close()


def most_common_element(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    if most_common:
        return most_common[0][0]
    else:
        return None


def repair_qq_patch(input_class=None, output_class=None, num_rows=None, num_colums=None):
    num_rows = num_rows - 1
    num_colums = num_colums - 1

    key_value_3x3 = {}
    for r in range(1, num_rows):
        for c in range(1, num_colums):
            key = str(r) + "_" + str(c)
            value = [[r - 1, c - 1], [r - 1, c], [r - 1, c + 1], [r, c - 1], [r, c + 1], [r + 1, c - 1], [r + 1, c],
                     [r + 1, c + 1]]
            key_value_3x3[key] = value

    center_key_value = {}
    with open(input_class, 'r') as file:
        for line in file:
            content = line.strip()
            key = content.split("+")[0]
            value = content.split(" --> ")[1]
            center_key_value[key] = value

    count = 0
    with open(output_class, 'w', encoding='utf-8') as file:
        for key in center_key_value.keys():
            r = int(key.split("_")[0])
            c = int(key.split("_")[1])
            if r == 0:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
                continue
            if r == num_rows:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
                continue
            if c == 0:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
                continue
            if c == num_colums:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
                continue
            count += 1
            value_8_patch = key_value_3x3[key]
            center_patch_lable = center_key_value[key]  # 中心图像标签
            surrounding_patch_lable = []
            for patch_id in value_8_patch:
                id_key = str(patch_id[0]) + "_" + str(patch_id[1])
                surrounding_patch_lable.append(center_key_value[id_key])

            count_dict = Counter(surrounding_patch_lable)  # 对列表里面重复出现的元素个数进行次数统计，返回dict

            if center_patch_lable != "Port" and center_patch_lable != "Airport":
                port_num = count_dict["Port"]
                airport_num = count_dict["Airport"]
                if airport_num > port_num and airport_num >= 5:
                    print("将" + key + "修改为：", "Airport")
                    file.write(key + "+Category.tif --> " + "Airport" + '\n')  # 在每行的末尾添加换行符
                elif port_num > airport_num and port_num >= 5:
                    print("将" + key + "修改为：", "Port")
                    file.write(key + "+Category.tif --> " + "Port" + '\n')  # 在每行的末尾添加换行符
                else:
                    file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
            else:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符

    file.close()


def repair_zy_patch(input_class=None, output_class=None, num_rows=None, num_colums=None):
    num_rows = num_rows - 1
    num_colums = num_colums - 1

    key_value_3x3 = {}
    for r in range(1, num_rows):
        for c in range(1, num_colums):
            key = str(r) + "_" + str(c)
            value = [[r - 1, c - 1], [r - 1, c], [r - 1, c + 1], [r, c - 1], [r, c + 1], [r + 1, c - 1], [r + 1, c],
                     [r + 1, c + 1]]
            key_value_3x3[key] = value

    center_key_value = {}
    with open(input_class, 'r') as file:
        for line in file:
            content = line.strip()
            key = content.split("+")[0]
            value = content.split(" --> ")[1]
            center_key_value[key] = value

    count = 0
    with open(output_class, 'w', encoding='utf-8') as file:
        for key in center_key_value.keys():
            r = int(key.split("_")[0])
            c = int(key.split("_")[1])
            if r == 0:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
                continue
            if r == num_rows:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
                continue
            if c == 0:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
                continue
            if c == num_colums:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符
                continue
            count += 1
            value_8_patch = key_value_3x3[key]
            center_patch_lable = center_key_value[key]  # 中心图像标签
            surrounding_patch_lable = []
            for patch_id in value_8_patch:
                id_key = str(patch_id[0]) + "_" + str(patch_id[1])
                surrounding_patch_lable.append(center_key_value[id_key])

            count_dict = Counter(surrounding_patch_lable)  # 对列表里面重复出现的元素个数进行次数统计，返回dict

            if center_patch_lable == "Airport":
                if "Airport" not in surrounding_patch_lable:
                    print("将" + key + "修改为：", most_common_element(surrounding_patch_lable))
                    file.write(
                        key + "+Category.tif --> " + most_common_element(surrounding_patch_lable) + '\n')  # 在每行的末尾添加换行符
                elif count_dict["Airport"] <= 2:
                    print("将" + key + "修改为：", most_common_element(surrounding_patch_lable))
                    file.write(
                        key + "+Category.tif --> " + most_common_element(surrounding_patch_lable) + '\n')  # 在每行的末尾添加换行符
                else:
                    file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符

            elif center_patch_lable == "Port":
                if "Port" not in surrounding_patch_lable:
                    print("将" + key + "修改为：", most_common_element(surrounding_patch_lable))
                    file.write(
                        key + "+Category.tif --> " + most_common_element(surrounding_patch_lable) + '\n')  # 在每行的末尾添加换行符
                else:
                    file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符

            elif center_patch_lable != "Port" and center_patch_lable != "Airport":
                port_num = count_dict["Port"]
                airport_num = count_dict["Airport"]
                if airport_num > port_num and airport_num >= 5:
                    print("将" + key + "修改为：", "Airport")
                    file.write(key + "+Category.tif --> " + "Airport" + '\n')  # 在每行的末尾添加换行符
                elif port_num > airport_num and port_num >= 5:
                    print("将" + key + "修改为：", "Port")
                    file.write(key + "+Category.tif --> " + "Port" + '\n')  # 在每行的末尾添加换行符
                else:
                    file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符

            else:
                file.write(key + "+Category.tif --> " + center_key_value[key] + '\n')  # 在每行的末尾添加换行符

    file.close()


def cut_roi_from_large_pic(is_project=False, save_category=None, pred_label_save_txt_post_processing=None,
                           pic_name=None, patch_size=None,
                           num_rows=None, num_colums=None):
    with open(pred_label_save_txt_post_processing, 'r') as file:
        lines = file.readlines()

    rows = []
    colums = []
    for line in lines:
        rc = line.strip().split(" --> ")[0].split("+")[0].split("_")
        category = line.strip().split(" --> ")[1]
        if category == save_category:
            if int(rc[0]) != num_rows - 1:
                rows.append(int(rc[0]))
            if int(rc[1]) != num_colums - 1:
                colums.append(int(rc[1]))

    rows.sort()
    colums.sort()

    # 打开原始图像
    image = Image.open(pic_name + ".tif")

    # 定义裁剪区域的坐标
    if is_project is True:
        x1, y1, x2, y2 = 0, 0, colums[-1], rows[-1]
    else:
        x1, y1, x2, y2 = colums[0], rows[0], colums[-1], rows[-1]
        # print(x1, y1, x2, y2)
    if x2 + 1 != num_colums:
        x2 += 1
    if y2 + 1 != num_rows:
        y2 += 1
    print(x1, y1, x2, y2)

    # 裁剪图像
    cropped_image = image.crop((x1 * patch_size, y1 * patch_size, x2 * patch_size, y2 * patch_size))

    save_img_path = pic_name + "_" + save_category + "_roi_post_processing.tif"

    # 保存裁剪后的图像
    cropped_image.save(save_img_path)

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cropped_image = cv2.cvtColor(np.asarray(cropped_image), cv2.COLOR_RGB2BGR)

    return save_img_path, image, cropped_image, x1 * patch_size, y1 * patch_size, x2 * patch_size, y2 * patch_size


import base64


def image_to_base64(img):
    # 将图像转换为JPEG格式
    _, buffer = cv2.imencode('.jpg', img)
    # 将图像数据转换为base64编码的字符串
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def scene_classification(pic_name=None, num_rows=None, ori_path=None):
    global json_file_path
    PIL_img_dict = {}
    num_colums = 24  # 要恢复的图像列数（从1开始）
    patch_size = 408
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ori_paths = ori_path
    large_image = Image.open(ori_paths)

    # 创建一个文件夹用于保存小图
    output_patches_folder = current_dir + "/patch_images_" + pic_name
    output_folder = output_patches_folder + '_3x3_cat'
    output_folder_surrounding = output_patches_folder + '_3x3_cat'

    pred_label_save_txt = current_dir + '/pred_' + pic_name + '_scene_class.txt'
    pred_label_save_txt_post_processing = current_dir + r'/pred_' + pic_name + '_scene_class_post_processing.txt'
    # 指向需要遍历预测的图像文件夹
    imgs_root = output_folder
    # read class_indict
    json_path = current_dir + '/run_RSWSC_Whole_Project/class_indices.json'
    # load model weights
    weights_path = current_dir + "/run_RSWSC_Whole_Project/weights/model_best4.pth"

    print("开始裁剪图像...")
    large_patches(pic_name, current_dir, large_image, output_patches_folder, patch_size)
    print("裁剪图像完成！")
    # print()
    print("开始连接图像...")
    cat_imgs_inner_and_surrounding(output_folder, output_folder_surrounding, output_patches_folder)
    print("连接图像完成！")
    # print()
    print("开始预测图像...")
    run_model(pic_name, pred_label_save_txt, imgs_root, json_path, weights_path)
    print("预测图像完成！")
    # print()

    if pic_name == 'qq':
        print("后处理qq...")
        repair_qq_patch(input_class=pred_label_save_txt, output_class=pred_label_save_txt_post_processing,
                        num_rows=num_rows, num_colums=num_colums)
        print("qq图像后处理！")
        # print()
        print("开始保存图像...")
        save_img_path, image, cropped_image, x1, y1, x2, y2 = cut_roi_from_large_pic(save_category="Airport",
                                                                                     pred_label_save_txt_post_processing=pred_label_save_txt_post_processing,
                                                                                     pic_name=pic_name,
                                                                                     patch_size=patch_size,
                                                                                     num_rows=num_rows,
                                                                                     num_colums=num_colums)
        # PIL_img_dict["Airport_Name"] = [save_img_path]
        PIL_img_dict["Ori_Pic"] = [image_to_base64(image)]
        PIL_img_dict["ori_path"] = ori_path
        PIL_img_dict["Airport_ROI_Pic"] = [image_to_base64(cropped_image)]
        PIL_img_dict["Airport_ROI_Pos"] = [x1, y1, x2, y2]
        PIL_img_dict['name'] = pic_name
        PIL_img_dict["Port"] = []
        print("保存图像完成！")
        postfix = 'json'
        json_filename = f"{pic_name}.{postfix}"
        json_output_dir = '../result_json/'
        json_file_path = os.path.join(json_output_dir, json_filename)
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)
        PIL_img_dict['json_file_path'] = json_file_path
        json_data = json.dumps(PIL_img_dict, indent=4)
        with open(json_file_path, "w") as file:
            file.write(json_data)


    elif pic_name == 'zy':
        print("后处理zy...")
        repair_zy_patch(input_class=pred_label_save_txt, output_class=pred_label_save_txt_post_processing,
                        num_rows=num_rows, num_colums=num_colums)
        print("zy图像后处理！")
        # print()
        print("开始保存图像...")
        save_img_path1, image, cropped_image, x1, y1, x2, y2 = cut_roi_from_large_pic(save_category="Airport",
                                                                                      pred_label_save_txt_post_processing=pred_label_save_txt_post_processing,
                                                                                      pic_name=pic_name,
                                                                                      patch_size=patch_size,
                                                                                      num_rows=num_rows,
                                                                                      num_colums=num_colums)
        PIL_img_dict["Airport_Name"] = [save_img_path1]
        PIL_img_dict["Ori_Pic"] = [image_to_base64(image)]
        PIL_img_dict["ori_path"] = ori_path
        PIL_img_dict["Airport_ROI_Pic"] = [image_to_base64(cropped_image)]
        PIL_img_dict["Airport_ROI_Pos"] = [x1, y1, x2, y2]
        print("保存图像完成！")
        # print()
        print("开始保存图像...")
        save_img_path2, image, cropped_image, x1, y1, x2, y2 = cut_roi_from_large_pic(save_category="Port",
                                                                                      is_project=True,
                                                                                      pred_label_save_txt_post_processing=pred_label_save_txt_post_processing,
                                                                                      pic_name=pic_name,
                                                                                      patch_size=patch_size,
                                                                                      num_rows=num_rows,
                                                                                      num_colums=num_colums)
        PIL_img_dict["Port_Name"] = [save_img_path2]
        PIL_img_dict["Ori_path"] = [image_to_base64(image)]
        PIL_img_dict["Port_ROI_Pic"] = [image_to_base64(cropped_image)]
        PIL_img_dict["Port_ROI_Pos"] = [x1, y1, x2, y2]
        PIL_img_dict['name'] = pic_name
        print("保存图像完成！")
        # print()
        postfix = 'json'
        json_filename = f"{pic_name}.{postfix}"
        json_output_dir = '../result_json/'
        json_file_path = os.path.join(json_output_dir, json_filename)
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)
        PIL_img_dict['json_file_path'] = json_file_path
        json_data = json.dumps(PIL_img_dict, indent=4)

        with open(json_file_path, "w") as file:
            file.write(json_data)


    return  PIL_img_dict


from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)


# @app.route('/classify', methods=['GET', 'POST'])
# def class_begin():
#     # 处理跨域请求
#     if request.method == 'OPTIONS':
#         # 处理 OPTIONS 请求
#         response_headers = {
#             'Access-Control-Allow-Origin': '*',
#             'Access-Control-Allow-Methods': 'POST',
#             'Access-Control-Allow-Headers': 'Content-Type'
#         }
#         return ('', 204, response_headers)
#
#     pic_name = request.args.get("pic_name")
#     num_rows = request.args.get("num_rows")
#     ori_path = request.args.get("ori_path")
#     # pic_name = "zy"
#     # num_rows = 29  # 要恢复的图像行数（从1开始）  qq:32  zy:29
#     PIL_img_dict = scene_classification(pic_name, int(num_rows), ori_path)
#     # print("最终结果：", PIL_img_dict)
#
#     return jsonify(PIL_img_dict)
@app.route('/classify', methods=['GET', 'POST'])
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

    pic_name = request.args.get("pic_name")
    num_rows = request.args.get("num_rows")
    ori_path = request.args.get("ori_path")

    postfix = 'json'
    json_filename = f"{pic_name}.{postfix}"
    json_output_dir = './result_json/'
    json_file_path = os.path.join(json_output_dir, json_filename)
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return jsonify(data)
    else:
        PIL_img_dict = scene_classification(pic_name, int(num_rows), ori_path)
        return jsonify(PIL_img_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)

# http://127.0.0.1:5000/classify?weights_type=2&pic_name=zy&num_rows=29

# 输入的数据以及输入的格式，
# pic_name 输入的图片名称
# num_rows 输入的图片的行数
# num_colums 输入的图片的列数

# 输出的结果
# Ori_pic 原始大图
# Airport_ROI_Pic 裁剪的机场场景图
# Port_ ROI_Pic 裁剪的港口场景图
# Airport_ROI_Pos 裁剪的机场场景图在原图的左上角和右下角位置
# Port_ROI_Pos 裁剪的港口场景图在原图的左上角和右下角位置
