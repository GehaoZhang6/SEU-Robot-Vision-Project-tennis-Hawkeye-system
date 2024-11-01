import cv2
import numpy as np
import glob

# 找棋盘格角点
# 棋盘格模板规格(内角点个数，内角点是和其他格子连着的点,如10 X 7)
w = 11
h = 8

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp[:, :2]=objp[:, :2]
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

# 标定所用图像（路径不能有中文）
images = glob.glob( r"C:\Users\86152\Desktop\calibration\calibration\*.bmp")
names=[]
size = tuple()
for fname in images:
    # print(fname)
    img = cv2.imread(fname)
    print(fname)
    # 修改图像尺寸，参数依次为：输出图像，尺寸，沿x轴，y轴的缩放系数，INTER_AREA在缩小图像时效果较好
    # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
    # top_pad = 40
    # bottom_pad = 150
    #
    # # Add padding
    # gray =cv2.copyMakeBorder(gray, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    print(gray.shape)
    size = gray.shape[::-1]  # 矩阵转置

    # 找到棋盘格角点
    # 棋盘图像(8位灰度或彩色图像)  棋盘尺寸  存放角点的位置
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

    if ret==False:
        names.append(fname)
        continue
    # 角点精确检测
    # criteria:角点精准化迭代过程的终止条件(阈值)

    # cv2.TERM_CRITERIA_EPS
    # 表示当迭代的误差变化小于指定精度时停止。

    # cv2.TERM_CRITERIA_MAX_ITER
    # 表示当迭代次数达到指定的最大次数时停止
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100000, 0.000001)

    # 执行亚像素级角点检测 根据图像角点大小和密集程度调整窗口大小参数
    corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

    objpoints.append(objp)
    imgpoints.append(corners2)

    # 将角点在图像上显示
    cv2.drawChessboardCorners(img, (w, h), corners2, ret)
    cv2.imshow('findCorners', img)
    # cv2.imwrite('in.png',img)
    cv2.waitKey(10)
objpoints=np.asarray(objpoints)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
# print('objpoints',objpoints)
# print('imgpoints',imgpoints)
# print("ret:", ret)
print("内参数矩阵:\n", mtx, '\n')
print("畸变系数:\n", dist, '\n')
# print("旋转向量(外参数):\n", rvecs, '\n')
# print("平移向量(外参数):\n", tvecs, '\n')

intrinsic_camera_params = {
    "camera_7_1R": {
        "intrinsic_matrix": mtx,
        "distortion_coefficients":dist
    }
}
# 保存外参数据到.npy文件
# np.save("./camera_params/7_1R.npy", intrinsic_camera_params)

# 通过反投影误差，我们可以来评估结果的好坏。越接近0，说明结果越理想。
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("total error: ", total_error / len(objpoints))
print(names)