from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import cv2
# image_path = r"C:\Users\Administrator\Desktop\yaogan\qingquan\out\a.tif"
def draw_longitude(svimg,image_path,img_with_boxes):
    ds = gdal.Open(image_path)
    # image = cv2.imread(image_path)
    # 获取影像的地理转换信息和投影信息
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    min_x = geotransform[0]
    max_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    # 获取影像宽度和高度
    width = ds.RasterXSize
    height = ds.RasterYSize
    #将width和height换成点的坐标，相当于对点的位置进行了转换
    max_x = geotransform[0] + (width * geotransform[1] + height * geotransform[2])
    min_y = geotransform[3] + (width * geotransform[4] + height * geotransform[5])


    # 定义投影坐标系和经纬度坐标系的转换
    geodetic = pyproj.Proj(init='epsg:4326')  # 经纬度坐标系的 EPSG 编码

    # 将投影坐标转换为经纬度坐标 目标框中心点的坐标需要从pyproj.transform来调用
    lon_min_x, lat_min_y = pyproj.transform(projection, geodetic, min_x, min_y)
    lon_max_x, lat_max_y = pyproj.transform(projection, geodetic, max_x, max_y)

    # 将得到的经纬度坐标转换为度分秒格式
    def decimal_degrees_to_dms(dd,n=False):
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
        lon=str(degrees)+"°"+str(minutes)+"′"+str(seconds)+"″"
        # lat=str(degrees1)+"°"+str(minutes1)+"′"+str(seconds1)+"″"+"N"

        return lon

    # 绘制五等分经纬度格网
    num_divisions = 5  # 将影像五等分
    step_x = int(width/ num_divisions)
    step_y = int(height/ num_divisions)
    x_divisions = np.linspace(0, width, num_divisions + 1,dtype=int)
    y_divisions = np.linspace(0, height, num_divisions + 1,dtype=int)

    lon_x_divisions = np.linspace(lon_min_x, lon_max_x, num_divisions + 1)
    lat_y_divisions = np.flipud(np.linspace(lat_min_y, lat_max_y, num_divisions + 1))


    plt.figure(figsize=(10, 10))
    # plt.imshow(ds.ReadAsArray().transpose(1,2,0))
    plt.imshow(img_with_boxes)
    x_labels=[]
    y_labels=[]

    for i in range(1, num_divisions):
        plt.axvline(x=step_x * i, color='red')  # 垂直线
        plt.axhline(y=step_y * i, color='red')  # 水平线
    # 标识经纬度坐标
    for x,y in zip(lon_x_divisions,lat_y_divisions):
        # plt.axvline(x=x, color='red', linestyle='-')
        x2=decimal_degrees_to_dms(x, True)
        y2 = decimal_degrees_to_dms(y, True)

        x_labels.append(x2+"E")
        y_labels.append(y2 + "N")

    plt.xticks(x_divisions[1:-1], x_labels[1:-1],ha='center',fontsize=12)
    plt.yticks(y_divisions[1:-1] ,y_labels[1:-1],rotation=90,verticalalignment='center',fontsize=12)

    plt.title('Geographic Grid on Image',fontsize=15,pad=10)
    plt.tight_layout()
    plt.savefig(svimg,dpi=1000)
    # 读取保存的图像
    img = cv2.imread(svimg)

    combinate_name = image_path.split(".")[0]+'_'+'combinate'+'.tif'
    cv2.imwrite(combinate_name,img)
    # plt.show()
    return img, img.shape[1], img.shape[0]
