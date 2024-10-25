def inference(weights_path, img_path, crop_size, overlap_size, savejson_func, json_output_dir, postfix,
              Use_Masks, ds_min_x, ds_max_y, geotransform, projection, cut_pic_lefttop_point, num_id, conf_score):
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
    names = {0: 'oiltank', 1: 'wharf'}
    # 改舰船的标签名字
    if weights_path == 'weights/ship801.pt':
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
    elif weights_path == 'weights/airplane729.pt':
        names = {
            0: 'warplane_SU-35',
            1: 'transport_plane_C-130',
            2: 'transport_plane_C-17',
            3: 'transport_plane_C-5',
            4: 'warplane_F-16',
            5: 'bomber_TU-160',
            6: 'early_warning_airplane_E-3',
            7: 'bomber_B-52',
            8: 'antisubmarine_plane_P-3C',
            9: 'bomber_B-1B',
            10: 'transport_plane_E-8',
            11: 'bomber_TU-22',
            12: 'warplane_F-15',
            13: 'transport_plane_KC-135',
            14: 'warplane_F-22',
            15: 'attack_plane_FA,-18',
            16: 'bomber_TU-95',
            17: 'oil_plane_KC-10',
            18: 'warplane_SU-34',
            19: 'warplane_SU-24',
        }
    elif weights_path == 'weights/airport722.pt':
        names = {0: 'TBuild', 1: 'Taxiway', 2: 'Apron', 3: 'Runway', 4: 'Hangar', 5: 'Landpad'}
    elif weights_path == 'weights/harbor.pt':
        names = {0: 'oiltank', 1: 'wharf'}
    category_counts = {key: 0 for key in names}  # 类别计数
    # ————————————————————————————————————杨——————————————————————————————————#
    x_steps = range(0, width, crop_size[0] - overlap_size)
    y_steps = range(0, height, crop_size[1] - overlap_size)
    # detect_yes = False
    detected_coordinates = []  # 创建列表来保存检测到

    if weights_path == 'weights/airport722.pt':
        detected_coordinates = []
    if weights_path == 'weights/airplane729.pt':
        detected_coordinates = []
    if weights_path == 'weights/harbor.pt':
        detected_coordinates = []