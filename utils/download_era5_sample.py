import cdsapi

# 初始化客户端
c = cdsapi.Client()

# -----------------------------------------------------------
# 请根据上一步打印的结果，修改下面的变量！
# -----------------------------------------------------------
TARGET_YEAR = '2026'
TARGET_MONTH = '01'
TARGET_DAY = '01'
TARGET_TIME = '00:00'  # 请修改为离你掩星时间最近的整点 (例如 12:04 -> 12:00)

# 定义下载区域 (Bounding Box)
# 格式: [North, West, South, East]
# 为了保险，我们下载目标经纬度周围 +/- 0.5 度的范围
# 例如：如果 Lat=20.5, Lon=120.5
# North=21, South=20, West=120, East=121
center_lat = -25.26     # 纬度
center_lon = -107.76    # 经度
area = [
    center_lat + 0.5, 
    center_lon - 0.5, 
    center_lat - 0.5, 
    center_lon + 0.5
]

print(f"正在请求 ERA5 数据...")
print(f"时间: {TARGET_YEAR}-{TARGET_MONTH}-{TARGET_DAY} {TARGET_TIME}")
print(f"区域: {area}")

c.retrieve(
    'reanalysis-era5-pressure-levels', # 数据集名称
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',            # 必须是 netcdf
        'variable': [
            'temperature',             # 温度 (T)
            'specific_humidity',       # 比湿 (Q)
            'geopotential',            # 位势 (用于计算几何高度 Z)
        ],
        'pressure_level': [
            # 我们需要从地面到高空的所有层
            '1', '2', '3', '5', '7', '10',
            '20', '30', '50', '70', '100',
            '125', '150', '175', '200', '225',
            '250', '300', '350', '400', '450',
            '500', '550', '600', '650', '700',
            '750', '775', '800', '825', '850',
            '875', '900', '925', '950', '975',
            '1000',
        ],
        'year': TARGET_YEAR,
        'month': TARGET_MONTH,
        'day': TARGET_DAY,
        'time': TARGET_TIME,
        'area': area, # 下载小区域，速度快
    },
    'era5_sample.nc') # 保存的文件名

print("下载完成！文件保存为 era5_sample.nc")