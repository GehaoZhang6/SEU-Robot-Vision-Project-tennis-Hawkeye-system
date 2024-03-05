import cv2
import numpy as np
from kalman_filter import Kalman_filter
from box import BBox, BBoxViewer
import glob

save_trail=[]
class Detect_feature_track(object):
    def __init__(self, frame,foreground_image_list,background_image_list,mov_avg_window=10):

        self.params = cv2.SimpleBlobDetector_Params()
        self.params.minThreshold = 10
        self.params.maxThreshold = 230

        self.params.filterByColor = True
        self.params.blobColor = 255

        self.params.filterByArea = True
        self.params.minArea = 1
        self.params.maxArea = 10

        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.frame = frame
        self.original_frame=frame
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=50, detectShadows=False)

        self.kf = Kalman_filter()

        self.tick = 0

        self.prev_tick = 0

        self.sift = cv2.SIFT_create()

        self.bbox_size_hist = []
        self.mov_avg_window = mov_avg_window
        self.bbox = BBox((0, 0), (0, 0), self.frame.shape)
        for i in range(self.mov_avg_window):
            self.bbox_size_hist.append(self.bbox.br - self.bbox.tl)

        self.foreground_dsc = []
        self.background_dsc = []

        self.found_past = False
        self.not_found_count = 0
        self.count=0
        self.trail=[]

        self._initKeypoints(foreground_image_list, background_image_list)


    def _initKeypoints(self, foreground_image_list, background_image_list):
        print('initKeypoints')
        # 处理前景图像列表
        for foreground_path in foreground_image_list:
            foreground_frame = cv2.imread(foreground_path)
            if foreground_frame is not None:  # 检查图像是否成功读取
                foreground_frame_gray = cv2.cvtColor(foreground_frame, cv2.COLOR_BGR2GRAY)
                corners = self.detector.detect(foreground_frame_gray)
                if corners is not None:
                    _, foreground_descriptors = self.sift.compute(foreground_frame_gray, corners)
                if foreground_descriptors is not None:
                    for descriptor in np.asarray(foreground_descriptors):
                        self.foreground_dsc.append(descriptor)



        for background_path in background_image_list:
            background_frame = cv2.imread(background_path)
            if background_frame is not None:  # 检查图像是否成功读取
                background_frame_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
                corners = self.detector.detect(background_frame)
                if corners is not None:
                    _, background_descriptors = self.sift.compute(background_frame_gray, corners)
                if background_descriptors is not None:
                    for descriptor in np.asarray(background_descriptors):
                        self.background_dsc.append(descriptor)

        self.foreground_dsc = np.asarray(self.foreground_dsc)
        self.background_dsc = np.asarray(self.background_dsc)


    def _match(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        corners = self.detector.detect(gray)

        if corners is not None:
            for corner in corners:
                x, y = corner.pt
                x, y = int(x), int(y)
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            keypoints, descriptors = self.sift.compute(gray, corners)

            return 1,keypoints, descriptors
        else:
            return 0,0,0

    def _detet_foreground(self):
        check,keypoints,descriptors=self._match()
        found = False
        if check :

            draw_foreground_kpt=[]
            max_distance = 100
            max_distance_id = 0
            bf = cv2.BFMatcher()

            fore_group = bf.knnMatch(descriptors, self.foreground_dsc, k=3)

            back_group = bf.knnMatch(descriptors, self.background_dsc, k=3)

            if fore_group is not None:
                for m1, m2 in zip(fore_group, back_group):
                    # print(m1)
                    # print(m1[0])
                    if m2[0].distance!=0 and (m1[0].distance/m2[0].distance)<=1 and (m1[1].distance/m2[1].distance)<=0.95\
                            and (m1[2].distance/m2[2].distance)<=0.85:
                        if (m1[2].distance/m2[2].distance)<max_distance:
                            max_distance=(m1[0].distance/m2[0].distance)
                            max_distance_id=m1[0].queryIdx
                    elif m2[0].distance!=0 and (m1[0].distance/m2[0].distance)>=1.1:
                        self.background_dsc = np.delete(self.background_dsc, 0, axis=0)
                        self.background_dsc = np.vstack([self.background_dsc, descriptors[m1[0].queryIdx]]).astype(
                            np.float32)
                if  max_distance!=100:
                    self.foreground_dsc = np.delete(self.foreground_dsc, 0, axis=0)
                    self.foreground_dsc = np.vstack([self.foreground_dsc, descriptors[max_distance_id]]).astype(np.float32)
                    self.trail.append(keypoints[max_distance_id])
                    save_trail.append(keypoints[max_distance_id].pt)
                    draw_foreground_kpt.append(keypoints[max_distance_id].pt)
                else:
                    save_trail.append([0,0])


                img_with_keypoints = cv2.drawKeypoints(self.frame, self.trail, None, color=(0, 0, 255))

                cv2.namedWindow('Image with Keypoints', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Image with Keypoints', 600, 400)
                cv2.imshow('Image with Keypoints', img_with_keypoints)
                cv2.waitKey(10)

                if len(draw_foreground_kpt)!=0:
                    draw_foreground_kpt = np.array(draw_foreground_kpt).astype(np.float32)
                    found = True
                    self.found_past=True
                    rect = cv2.boundingRect(draw_foreground_kpt)
                    center_kpts = np.mean(draw_foreground_kpt, axis=0)
                    return found,rect,center_kpts
        return found,0,0

    def _get_mask(self,frame):
        self.count+=1
        kernel_size = (5, 5)
        sigma_x = 0
        sigma_y = 0
        frame = cv2.GaussianBlur(frame, kernel_size, sigma_x, sigma_y)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[cv2.medianBlur(self.fgbg.apply(frame), ksize=9) == 0] = 0
        kernel = np.ones((5, 5), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=9)
        mask_result = cv2.bitwise_or(frame, frame, mask=gray)

        self.frame = mask_result
    def track(self,frame):
        self._get_mask(frame)
        self.prev_tick = self.tick
        self.tick = cv2.getTickCount()
        dT = (self.tick - self.prev_tick) / cv2.getTickFrequency()

        if self.found_past:
            self.kf.deltaTime(dT)
            state = self.kf.kf.predict()

            center = np.array([0, 0])
            window_size = np.array([0, 0])

            center[0] = state[0]
            center[1] = state[1]

            window_size[0] = state[4]
            window_size[1] = state[5]

            p1 = np.array(center - window_size / 2)
            p2 = np.array(center + window_size / 2)

            self.bbox.update(p1, p2)
            self.bbox_size_hist.pop(0)
            self.bbox_size_hist.append(window_size)

        found, rect, center_kpts = self._detet_foreground()
        if not found:
            self.not_found_count += 1
            if self.not_found_count > 50:
                self.found_past = False
        else:

            self.not_found_count = 0
            x, y, w, h = rect
            center_bounding = np.array([x + w / 2, y + h / 2])
            center = 0.7 * center_kpts + 0.3 * center_bounding

            w_avg, h_avg = tuple(np.mean(np.asarray(self.bbox_size_hist), axis=0))
            window_size = np.array([0, 0])
            window_size[0] = 0.5 * w_avg + 0.5 * w
            window_size[1] = 0.5 * h_avg + 0.5 * h
            measure = np.array([center[0], center[1], window_size[0], window_size[1]], dtype=np.float32)

            if not self.found_past:
                self.kf.reset(measure)
                self.found_past = True
            else:
                self.kf.kf.correct(measure)

def read_images_with_glob(pattern):
    images = []
    for image_path in glob.glob(pattern):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            images.append(img)
        else:
            print(f"Error reading image: {image_path}")

    return images

foreground_pattern=r'./foreground/*.jpg'
foreground_image_list = glob.glob(foreground_pattern)
background_pattern=r'./background/*.png'
background_image_list = glob.glob(background_pattern)

frame = cv2.imread(r'./background/frame_0000.png', cv2.IMREAD_GRAYSCALE)
tracker = Detect_feature_track(frame, foreground_image_list,background_image_list)
viewer = BBoxViewer()

video_path = r'./videos_original/7_1L.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break
    tracker.track(frame)

    img = viewer.draw(tracker.bbox, frame)
    cv2.namedWindow('tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tracking', 600, 400)
    cv2.imshow('tracking', img)
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()
print(save_trail)
# np.save(r'./ball_1L.npy',save_trail)