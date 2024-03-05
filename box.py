import numpy as np
import cv2

class BBox(object):
    def __init__(self, p1, p2, size, padding = 20):
        self.padding = int(padding)
        self.size = size
        self.tl = np.array([int(float(p1[0])), int(float(p1[1]))])
        self.br = np.array([int(float(p2[0])), int(float(p2[1]))])
        self.search_all = True


    def update(self, p1, p2):
        self.tl = np.array([int(float(p1[0])), int(float(p1[1]))])
        self.br = np.array([int(float(p2[0])), int(float(p2[1]))])

    def inside(self, kpts):
        if (self.br[0] - self.tl[0] + self.br[1] - self.tl[1]) < 10:
            self.search_all = True
        if self.search_all:
            self.tl_search = np.array([0, 0])
            self.br_search = np.array([self.size[0] - 1, self.size[1] - 1])
        else:
            self.tl_search = np.array([self.tl[0] - self.padding, self.tl[1] - self.padding])
            self.br_search = np.array([self.br[0] + self.padding, self.br[1] + self.padding])
        result = []
        for kpt in kpts:
            result.append(self._ptInside(kpt.pt))
        return result

    def _ptInside(self, pt):
        logic = self.tl_search[0] <= pt[0] <= self.br_search[0] and self.tl_search[1] <= pt[1] <= self.br_search[1]
        return logic

class BBoxViewer(object):
    def draw(self, bbox, frame):
        img = frame.copy()
        center = np.array((bbox.tl + bbox.br) / 2, dtype=int)
        cv2.circle(img, tuple(center), 3, (0,0,255), -1)
        return img