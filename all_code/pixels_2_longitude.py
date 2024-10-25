import pyproj
import torch


# 将得到的经纬度坐标转换为度分秒格式
def decimal_degrees_to_dms(dd, n=True):
    degrees = int(dd)
    minutes = int((dd - degrees) * 60)
    if n is True:
        seconds = int(((dd - degrees) * 60 - minutes) * 60)
        # seconds1 = int(((ee - degrees) * 60 - minutes) * 60)
    else:
        seconds = ((dd - degrees) * 60 - minutes) * 60
        # seconds1 = ((ee - degrees) * 60 - minutes) * 60
    # degrees1 = int(ee)
    # minutes1 = int((ee - degrees) * 60)
    lon = str(degrees) + "°" + str(minutes) + "′" + str(seconds) + "″"
    # lat=str(degrees1)+"°"+str(minutes1)+"′"+str(seconds1)+"″"+"N"

    return lon
def pixels_2_longitude(ds_min_x,ds_max_y,geotransform,projection,x1,y1,x2,y2):
    # 定义投影坐标系和经纬度坐标系的转换
    geodetic = pyproj.Proj(init='epsg:4326')  # 经纬度坐标系的 EPSG 编码
    # 将投影坐标转换为经纬度坐标 目标框中心点的坐标需要从pyproj.transform来调用
    t_x1 = geotransform[0] + (x1 * geotransform[1] + y1 * geotransform[2])
    t_y1 = geotransform[3] + (x1 * geotransform[4] + y1 * geotransform[5])
    t_x1 = t_x1.cpu()
    t_y1 = t_y1.cpu()
    t_x1, t_y1 = pyproj.transform(projection, geodetic, t_x1, t_y1)

    t_x2 = geotransform[0] + (x2 * geotransform[1] + y2 * geotransform[2])
    t_y2 = geotransform[3] + (x2 * geotransform[4] + y2 * geotransform[5])
    t_x2 = t_x2.cpu()
    t_y2 = t_y2.cpu()
    t_x2, t_y2 = pyproj.transform(projection, geodetic, t_x2, t_y2)
    x_center_longitude = round((t_x1+t_x2)/2,5)
    y_center_longitude = round((t_y1+t_y2)/2,5)
    # 转成度分秒
    # tt_x1 = decimal_degrees_to_dms(x_center_longitude, True)
    # tt_y1 = decimal_degrees_to_dms(y_center_longitude, True)

    # return  tt_x1,tt_y1
    return x_center_longitude,y_center_longitude,t_x1, t_y1,t_x2, t_y2