# 定义字符串
s = '18°14‘2.8428”'

# 去掉不需要的字符
value = s.split('°')[1].split('‘')[0]

print(value)
