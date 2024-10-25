# -*- coding: gbk -*-
def find_max_in_nested_list(nested_list):
    max_value = float('-inf')  # ��ʼ��Ϊ�������
    for item in nested_list:
        if isinstance(item, list):  # ������б���ݹ����
            max_value = max(max_value, find_max_in_nested_list(item))
        else:  # ��������֣���Ƚϴ�С
            max_value = max(max_value, item)

    return max_value

# ʾ������
box_mask = [[3400, 2600, 3400], 2600]

# �������ֵ
max_number = find_max_in_nested_list(box_mask)
print("����������:", max_number)
