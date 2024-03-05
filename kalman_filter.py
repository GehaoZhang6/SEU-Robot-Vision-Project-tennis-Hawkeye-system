import numpy as np
import cv2

class Kalman_filter(object):
    def __init__(self):
        # 初始化Kalman滤波器，6表示状态向量的维度，4表示测量向量的维度，0表示控制向量的维度
        # 1，2:方框中心x,y坐标  3,4:        5,6:方框长，宽
        self.kf=cv2.KalmanFilter(6,4,0)

        # 设置状态转移矩阵为单位矩阵
        cv2.setIdentity(self.kf.transitionMatrix)

        # 设置测量矩阵，将状态向量映射到测量向量
        self.kf.measurementMatrix=np.zeros((4,6),dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 4] = 1
        self.kf.measurementMatrix[3, 5] = 1

        # 设置过程噪声协方差矩阵，表示状态转移时的噪声
        cv2.setIdentity(self.kf.processNoiseCov,1e-2)
        self.kf.processNoiseCov[2, 2] = 5
        self.kf.processNoiseCov[3, 3] = 5

        # 设置测量噪声协方差矩阵
        cv2.setIdentity(self.kf.measurementNoiseCov,1e-1)

    def deltaTime(self,dT):
        # 根据时间步长更新状态转移矩阵
        self.kf.transitionMatrix[0, 2] = dT
        self.kf.transitionMatrix[1, 3] = dT

    def reset(self,measure):
        # 重置Kalman滤波器的状态
        self.kf.errorCovPre[0, 0] = 1
        self.kf.errorCovPre[1, 1] = 1
        self.kf.errorCovPre[2, 2] = 1
        self.kf.errorCovPre[3, 3] = 1
        self.kf.errorCovPre[4, 4] = 1
        self.kf.errorCovPre[5, 5] = 1

        state = np.zeros(6, dtype=np.float32)
        state[0] = measure[0]
        state[1] = measure[1]
        state[4] = measure[2]
        state[5] = measure[3]

        # 设置Kalman滤波器的初始状态
        self.kf.statePost = state

