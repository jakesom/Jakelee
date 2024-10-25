# -*- coding: gbk -*-
def find_max_in_nested_list(nested_list):
    max_value = float('-inf')  # 初始化为负无穷大
    for item in nested_list:
        if isinstance(item, list):  # 如果是列表，则递归调用
            max_value = max(max_value, find_max_in_nested_list(item))
        else:  # 如果是数字，则比较大小
            max_value = max(max_value, item)

    return max_value

# 示例数据
box_mask = [[3400, 2600, 3400], 2600]

# 查找最大值
max_number = find_max_in_nested_list(box_mask)
print("最大的数字是:", max_number)
