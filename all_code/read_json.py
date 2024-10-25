import json

with open(r'./qq.json', 'r') as json_file:
    # 使用json.load()加载JSON数据
    all_data = json.load(json_file)
    print(all_data)