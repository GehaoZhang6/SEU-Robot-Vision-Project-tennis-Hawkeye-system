import cv2
import numpy as np

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', image)

        points.append((x, y))


image = cv2.imread('./extrinsic_pictures/1R_forPnP2.jpg')
cv2.imshow('image', image)

points = []

cv2.setMouseCallback('image', select_points)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

points_array = np.array(points)

np.save(r'./camera_params/camera_coordinate_1R.npy', points_array)

cv2.destroyAllWindows()
