import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

imgpoints_L=np.load(r'./ball_1L.npy',allow_pickle=True)
imgpoints_1L = np.asarray(imgpoints_L).astype(np.float32)
imgpoints_1L=imgpoints_1L[90:330,:]

imgpoints_R=np.load(r'./ball_1R.npy',allow_pickle=True)
imgpoints_1R = np.asarray(imgpoints_R).astype(np.float32)
imgpoints_1R=imgpoints_1R[90:330,:]

camera_matrix=np.load(r'./camera_params/1L_to_world.npy',allow_pickle=True).item()['1L']
camera_1L_K = np.asarray(camera_matrix)

camera_matrix=np.load(r'./camera_params/1R_to_world.npy',allow_pickle=True).item()['1R']
camera_1R_K = np.asarray(camera_matrix)

imgpoints_L=np.load(r'./ball_2L.npy',allow_pickle=True)
imgpoints_2L = np.asarray(imgpoints_L).astype(np.float32)
imgpoints_2L=imgpoints_2L[90:,:]

imgpoints_R=np.load(r'./ball_2R.npy',allow_pickle=True)
imgpoints_2R = np.asarray(imgpoints_R).astype(np.float32)
imgpoints_2R=imgpoints_2R[90:,:]

camera_matrix=np.load(r'./camera_params/2L_to_world.npy',allow_pickle=True).item()['2L']
camera_2L_K = np.asarray(camera_matrix)

camera_matrix=np.load(r'./camera_params/2R_to_world.npy',allow_pickle=True).item()['2R']
camera_2R_K = np.asarray(camera_matrix)

def is_zero(p: np.ndarray) -> bool:

    if p[0] == 0 and p[1] == 0:
        return True
    else:
        return False

def inside_range(point: np.ndarray) -> bool:

    return -1 < point[0] < 25 and -1 < point[1] < 13 and -0.1 < point[2] < 100

def remove_outlier(p3d,iteration=1):
    for i in range(iteration):
        samecount = 0
        for i in range(1, np.size(p3d, 0)):
            neighs = []
            j = 0
            while len(neighs) != 3:
                if i + j < len(p3d):
                    j += 1
                else:
                    break
                if not is_zero(p3d[i + j - 1, :]):
                    neighs.append(p3d[i + j - 1, :])
            if len(neighs) > 0:
                arr = np.array(neighs)
                means = np.mean(arr, axis=0)
                norm = np.linalg.norm(p3d[i, :] - means)
                if norm > 3 and samecount < 5:
                    samecount += 1
                    p3d[i, :] = 0
                else:
                    samecount = 0
    return p3d

p3d_1 = []
for j in range(imgpoints_1R.shape[0]):
    if is_zero(imgpoints_1R[j, :]) or is_zero(imgpoints_1L[j, :]):
        p3d_1.append(np.array([0, 0, 0]))
    else:
        point4d = cv2.triangulatePoints(camera_1L_K, camera_1R_K, imgpoints_1L[j, :].T,
                                         imgpoints_1R[j, :].T)

        point = (point4d[:3] / point4d[3]).T
        point=point.reshape(3,)
        if inside_range(point):
            p3d_1.append(point)
        else:
            p3d_1.append(np.array([0, 0, 0]))
p3d_1 = np.array(p3d_1)

p3d_2 = []
for j in range(imgpoints_2R.shape[0]):
    if is_zero(imgpoints_2R[j, :]) or is_zero(imgpoints_2L[j, :]):
        p3d_2.append(np.array([0, 0, 0]))
    else:
        point4d = cv2.triangulatePoints(camera_2L_K, camera_2R_K, imgpoints_2L[j, :].T,
                                         imgpoints_2R[j, :].T)

        point = (point4d[:3] / point4d[3]).T
        point=point.reshape(3,)
        if inside_range(point):
            p3d_2.append(point)
        else:
            p3d_2.append(np.array([0, 0, 0]))
p3d_2 = np.array(p3d_2)


p3d_1=remove_outlier(p3d_1,iteration=3)
p3d_2=remove_outlier(p3d_2,iteration=3)

p3d=[]
for i in range(p3d_1.shape[0]):
    p3d.append(p3d_1[i,:])
    p3d.append(p3d_2[i,:])
for i in range(p3d_1.shape[0],p3d_2.shape[0]):
    p3d.append(p3d_2[i, :])
p3d = np.array(p3d)


def knn_filter_outliers(points_3d, n_neighbors, deviation, plane, show=True):
    if plane not in ['yz', 'xy', 'xz']:
        raise ValueError("Invalid plane parameter. It should be one of: 'yz', 'xy', or 'xz'.")
    points_3d=points_3d.copy()
    points_2d = points_3d[:, :-1] if plane == 'xy' else points_3d[:, [0, 2]] if plane == 'xz' else points_3d[:, 1:]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(points_2d)
    distances, indices = nbrs.kneighbors(points_2d)

    outlier_scores = distances[:, -1]
    threshold = np.mean(outlier_scores) + deviation

    outliers = points_2d[outlier_scores > threshold]
    outlier_indices = np.where(outlier_scores > threshold)[0]

    points_3d[outlier_indices] = 0

    if show:
        plt.title("K Nearest Neighbors (KNN)")
        plt.scatter(points_2d[:, 0], points_2d[:, 1], color='k', s=3., label='Data points')
        plt.scatter(outliers[:, 0], outliers[:, 1], color='r', s=100, edgecolors='k', label='Outliers')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return points_3d

p3d=knn_filter_outliers(p3d,4,1,plane='xy',show=0)
p3d=knn_filter_outliers(p3d,5,0.6,plane='yz',show=0)

p3d = p3d[p3d[:, -1] != 0]

def find_bounce(p3d):
    z_limit_high = 1
    z_limit_low =0.3
    bounce_indices = []

    for i in range(len(p3d) - 4):
        points_z = p3d[i:i+5, -1]
        if points_z[0] > points_z[1] > points_z[2] and points_z[4] > points_z[3] > points_z[2] and z_limit_low < points_z[2] < z_limit_high:
            bounce_indices.append(i+2)
    return bounce_indices

def find_hit(p3d):
    hit_indices=[]
    for i in range(len(p3d) - 5):
        points_x = p3d[i:i+5, 0]
        if points_x[0] > points_x[1] > points_x[2] and points_x[4] > points_x[3] > points_x[2] or \
                points_x[0] < points_x[1] < points_x[2] and points_x[4] < points_x[3] < points_x[2]:
            hit_indices.append(i+2)
    return hit_indices

def find_service(p3d):
    service_indices=[]
    for i in range(len(p3d) - 4):
        points_x = p3d[i:i+5, 0]
        if (abs(points_x[0]) + abs(points_x[1]))*5 < (abs(points_x[4]) + abs(points_x[3])) :
            service_indices.append(i+2)
    return service_indices

def model_curve_function(params,x):

    return params[0] + params[1] * x + params[2] * x**2

def model_line_function(params,x):

    return params[0] + params[1] * x

initial_guess_curve = [1, 1, 1]
initial_guess_line = [1, 1]


def fit_curve_with_constraints(x_data, y_data ,model_func,initial_guess,insert_num=30,show=True):

    start_point=[x_data[0],y_data[0]]
    end_point=[x_data[-1],y_data[-1]]
    def loss_func(params, x, y, start_point, end_point):
        predicted_y = model_func(params, x)
        start_loss = (predicted_y[0] - start_point[1]) ** 2
        end_loss = (predicted_y[-1] - end_point[1]) ** 2
        data_loss = np.mean((predicted_y - y) ** 2)
        return start_loss + end_loss + data_loss

    result = minimize(loss_func, initial_guess, args=(x_data, y_data, start_point, end_point))

    params_opt = result.x

    x_fit = np.linspace(min(x_data), max(x_data), insert_num)
    y_fit = model_func(params_opt, x_fit)
    if show:
        plt.scatter(x_data, y_data, label='Original Data')
        plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    return params_opt, x_fit, y_fit


def fit_service(service_trace):
    service_trace_temp1 = knn_filter_outliers(service_trace, 4, 0, plane='yz', show=0)
    plane_yz = service_trace_temp1[service_trace_temp1[:, -1] != 0, 1:]
    step=(np.max(plane_yz[:, 0])-np.min(plane_yz[:,0]))/len(plane_yz)
    for i in range(len(plane_yz) - 1):
        plane_yz[i + 1, 0] = plane_yz[0,0] + step*i
    x_data, y_data = plane_yz[:, 0], plane_yz[:, 1]
    params_opt, y_fit, z_fit = fit_curve_with_constraints(x_data, y_data ,model_curve_function, initial_guess_curve,50,show=0)

    service_trace_temp2 = knn_filter_outliers(service_trace, 6, 0.5, plane='xy', show=0)
    plane_xy = service_trace_temp2[service_trace_temp2[:,-1]!=0, :-1]
    step = (np.max(plane_xy[:, 0]) - np.min(plane_xy[:, 0])) / len(plane_xy)
    for i in range(len(plane_xy) - 1):
        plane_xy[i + 1, 0] = plane_xy[0, 0] + step*i
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, y_fit, x_fit = fit_curve_with_constraints(y_data, x_data ,model_line_function, initial_guess_line,50,show=0)
    p3d=np.vstack([x_fit,y_fit,z_fit])

    return p3d.T

def fit_first_curve(first_curve):
    first_curve_temp1 = knn_filter_outliers(first_curve, 4, 2, plane='yz', show=0)
    plane_yz = first_curve_temp1[first_curve_temp1[:, -1] != 0, 1:]
    step=(np.max(plane_yz[:, 0])-np.min(plane_yz[:,0]))/len(plane_yz)
    for i in range(len(plane_yz) - 1):
        plane_yz[i + 1, 0] = plane_yz[0, 0] + step*i
    y_data, z_data = plane_yz[:, 0], plane_yz[:, 1]
    params_opt, y_fit, z_fit = fit_curve_with_constraints(y_data, z_data ,model_curve_function, initial_guess_curve,100,show=0)

    first_curve_temp2 = knn_filter_outliers(first_curve, 6, 5, plane='xy', show=0)
    plane_xy = first_curve_temp2[first_curve_temp2[:,-1]!=0, :-1]
    boundary_y=plane_xy[0,1]
    for i in range(len(plane_xy)):
        if plane_xy[i,1]<boundary_y:
            plane_xy[i, :]=0
    plane_xy = plane_xy[plane_xy[:, -1] != 0]
    step = (np.max(plane_xy[:, 0]) - np.min(plane_xy[:, 0])) / len(plane_xy)
    for i in range(len(plane_xy) - 1):
        plane_xy[i + 1, 0] = plane_xy[0, 0] + step*i
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, y_fit, x_fit = fit_curve_with_constraints(y_data, x_data ,model_line_function, initial_guess_line,100,show=0)
    p3d=np.vstack([x_fit,y_fit,z_fit])

    return p3d.T

def fit_second_curve1(second_curve1):
    second_curve1_temp1 = knn_filter_outliers(second_curve1, 3, 1.5, plane='xy', show=0)
    plane_xy = second_curve1_temp1[second_curve1_temp1[:,-1]!=0, :-1]
    step = (np.max(plane_xy[:, 0]) - np.min(plane_xy[:, 0])) / len(plane_xy)
    for i in range(len(plane_xy) - 1):
        plane_xy[i + 1, 0] = plane_xy[0, 0] + step*i
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, y_fit, x_fit = fit_curve_with_constraints(y_data, x_data ,model_line_function, initial_guess_line,show=0)

    second_curve1_temp2 = knn_filter_outliers(second_curve1, 3, 0.8, plane='yz', show=0)
    plane_yz = second_curve1_temp2[second_curve1_temp2[:, -1] != 0, 1:]
    step=(np.max(plane_yz[:, 0])-np.min(plane_yz[:,0]))/len(plane_yz)
    for i in range(len(plane_yz) - 1):
        plane_yz[i + 1, 0] = plane_yz[0, 0] + step*i
    y_data, z_data = plane_yz[:, 0], plane_yz[:, 1]
    params_opt, y_fit, z_fit = fit_curve_with_constraints(y_data, z_data ,model_curve_function, initial_guess_curve,show=0)

    p3d=np.vstack([x_fit,y_fit,z_fit])
    return p3d.T

def fit_second_curve2(second_curve2):
    second_curve2_temp1 = knn_filter_outliers(second_curve2, 4, 2, plane='xz', show=0)
    plane_xz = second_curve2_temp1[second_curve2_temp1[:, -1] != 0][:, [0, 2]]
    boundary_x=plane_xz[0,0]
    for i in range(len(plane_xz)):
        if plane_xz[i,0]>boundary_x:
            plane_xz[i, :]=0
    plane_xz=plane_xz[plane_xz[:, -1] != 0]
    x_data, z_data = plane_xz[:, 0], plane_xz[:, 1]
    params_opt, x_fit, z_fit = fit_curve_with_constraints(x_data, z_data ,model_curve_function, initial_guess_curve,100,show=0)

    second_curve2_temp2 = knn_filter_outliers(second_curve2, 6, 5, plane='xy', show=0)
    plane_xy = second_curve2_temp2[second_curve2_temp2[:,-1]!=0, :-1]
    boundary_y=plane_xy[0,1]
    for i in range(len(plane_xy)):
        if plane_xy[i,1]>boundary_y:
            plane_xy[i, :]=0
    plane_xy=plane_xy[plane_xy[:, -1] != 0]
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, x_fit, y_fit = fit_curve_with_constraints(x_data, y_data ,model_line_function, initial_guess_line,100,show=0)
    p3d=np.vstack([x_fit,y_fit,z_fit])
    return p3d.T

def fit_third_curve1(third_curve1):
    third_curve1_temp1 = knn_filter_outliers(third_curve1, 4, 2, plane='xz', show=0)
    plane_xz = third_curve1_temp1[third_curve1_temp1[:, -1] != 0][:, [0, 2]]
    boundary_x=plane_xz[0,0]
    for i in range(len(plane_xz)):
        if plane_xz[i,0]>boundary_x:
            plane_xz[i, :]=0
    plane_xz=plane_xz[plane_xz[:, -1] != 0]
    x_data, z_data = plane_xz[:, 0], plane_xz[:, 1]
    params_opt, x_fit, z_fit = fit_curve_with_constraints(x_data, z_data ,model_curve_function, initial_guess_curve,show=0)

    third_curve1_temp2 = knn_filter_outliers(third_curve1, 6, 5, plane='xy', show=0)
    plane_xy = third_curve1_temp2[third_curve1_temp2[:,-1]!=0, :-1]
    boundary_y=plane_xy[0,1]
    for i in range(len(plane_xy)):
        if plane_xy[i,1]>boundary_y:
            plane_xy[i, :]=0
    plane_xy=plane_xy[plane_xy[:, -1] != 0]
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, x_fit, y_fit = fit_curve_with_constraints(x_data, y_data ,model_line_function, initial_guess_line,show=0)
    p3d=np.vstack([x_fit,y_fit,z_fit])
    return p3d.T

def fit_third_curve2(third_curve2):
    third_curve2_temp1 = knn_filter_outliers(third_curve2, 4, 2, plane='xz', show=0)
    plane_xz = third_curve2_temp1[third_curve2_temp1[:, -1] != 0][:, [0, 2]]
    boundary_x=plane_xz[0,0]
    for i in range(len(plane_xz)):
        if plane_xz[i,0]<boundary_x:
            plane_xz[i, :]=0
    plane_xz=plane_xz[plane_xz[:, -1] != 0]
    x_data, z_data = plane_xz[:, 0], plane_xz[:, 1]
    params_opt, x_fit, z_fit = fit_curve_with_constraints(x_data, z_data ,model_curve_function, initial_guess_curve,70,show=0)

    third_curve2_temp2 = knn_filter_outliers(third_curve2, 6, 5, plane='xy', show=0)
    plane_xy = third_curve2_temp2[third_curve2_temp2[:,-1]!=0, :-1]
    boundary_y=plane_xy[0,1]
    for i in range(len(plane_xy)):
        if plane_xy[i,1]<boundary_y:
            plane_xy[i, :]=0
    plane_xy=plane_xy[plane_xy[:, -1] != 0]
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, x_fit, y_fit = fit_curve_with_constraints(x_data, y_data ,model_line_function, initial_guess_line,70,show=0)
    p3d=np.vstack([x_fit,y_fit,z_fit])

    return p3d.T

def fit_fourth_curve1(fourth_curve1):
    fourth_curve1_temp1 = knn_filter_outliers(fourth_curve1, 2, 0.2, plane='xy', show=0)
    plane_xy = fourth_curve1_temp1[fourth_curve1_temp1[:,-1]!=0, :-1]
    plane_xy=plane_xy[plane_xy[:, -1] != 0]
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, x_fit, y_fit = fit_curve_with_constraints(x_data, y_data ,model_line_function, initial_guess_line,show=0)

    fourth_curve1_temp2 = knn_filter_outliers(fourth_curve1, 3, 0.3, plane='xz', show=0)
    plane_xz = fourth_curve1_temp2[fourth_curve1_temp2[:, -1] != 0][:, [0, 2]]
    plane_xz=plane_xz[plane_xz[:, -1] != 0]
    x_data, z_data = plane_xz[:, 0], plane_xz[:, 1]
    params_opt, x_fit, z_fit = fit_curve_with_constraints(x_data, z_data ,model_curve_function, initial_guess_curve,show=0)

    p3d=np.vstack([x_fit,y_fit,z_fit])

    return p3d.T

def fit_fourth_curve2(fourth_curve2):
    indices_to_delete = np.where((fourth_curve2[:, 0] < 10) & (fourth_curve2[:, 1] > 3.5))
    fourth_curve2 = np.delete(fourth_curve2, indices_to_delete, axis=0)
    fourth_curve2_temp1 = knn_filter_outliers(fourth_curve2, 5, 5, plane='xy', show=0)
    plane_xy = fourth_curve2_temp1[fourth_curve2_temp1[:,-1]!=0, :-1]
    plane_xy=plane_xy[plane_xy[:, -1] != 0]
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, x_fit, y_fit = fit_curve_with_constraints(x_data, y_data ,model_line_function, initial_guess_line,50,show=0)

    fourth_curve2_temp2 = knn_filter_outliers(fourth_curve2, 3, 5, plane='xz', show=0)
    plane_xz = fourth_curve2_temp2[fourth_curve2_temp2[:, -1] != 0][:, [0, 2]]
    plane_xz=plane_xz[plane_xz[:, -1] != 0]
    x_data, z_data = plane_xz[:, 0], plane_xz[:, 1]
    params_opt, x_fit, z_fit = fit_curve_with_constraints(x_data, z_data ,model_curve_function, initial_guess_curve,50,show=0)

    p3d=np.vstack([x_fit,y_fit,z_fit])

    return p3d.T

def fit_fifth_curve1(fifth_curve1):
    indices_to_delete = np.where((fifth_curve1[:, 0] < 8) & (fifth_curve1[:, 1] > 3.8))
    fifth_curve1 = np.delete(fifth_curve1, indices_to_delete, axis=0)
    fifth_curve1_temp1 = knn_filter_outliers(fifth_curve1, 2, 0.2, plane='xy', show=0)
    plane_xy = fifth_curve1_temp1[fifth_curve1_temp1[:,-1]!=0, :-1]
    plane_xy=plane_xy[plane_xy[:, -1] != 0]
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, x_fit, y_fit = fit_curve_with_constraints(x_data, y_data ,model_line_function, initial_guess_line,show=0)

    fifth_curve1_temp2 = knn_filter_outliers(fifth_curve1, 3, 0.3, plane='xz', show=0)
    plane_xz = fifth_curve1_temp2[fifth_curve1_temp2[:, -1] != 0][:, [0, 2]]
    plane_xz=plane_xz[plane_xz[:, -1] != 0]
    x_data, z_data = plane_xz[:, 0], plane_xz[:, 1]
    params_opt, x_fit, z_fit = fit_curve_with_constraints(x_data, z_data ,model_curve_function, initial_guess_curve,show=0)

    p3d=np.vstack([x_fit,y_fit,z_fit])

    return p3d.T

def fit_fifth_curve2(fifth_curve2):

    fifth_curve2_temp1 = knn_filter_outliers(fifth_curve2, 2, 2, plane='xy', show=0)
    plane_xy = fifth_curve2_temp1[fifth_curve2_temp1[:,-1]!=0, :-1]
    plane_xy=plane_xy[plane_xy[:, -1] != 0]
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, x_fit, y_fit = fit_curve_with_constraints(x_data, y_data ,model_line_function, initial_guess_line,100,show=0)

    fifth_curve2_temp2 = knn_filter_outliers(fifth_curve2, 3, 2, plane='xz', show=0)
    plane_xz = fifth_curve2_temp2[fifth_curve2_temp2[:, -1] != 0][:, [0, 2]]
    plane_xz=plane_xz[plane_xz[:, -1] != 0]
    x_data, z_data = plane_xz[:, 0], plane_xz[:, 1]
    params_opt, x_fit, z_fit = fit_curve_with_constraints(x_data, z_data ,model_curve_function, initial_guess_curve,100,show=0)

    p3d=np.vstack([x_fit,y_fit,z_fit])

    return p3d.T

def fit_last_curve(last_curve):

    last_curve_temp1 = knn_filter_outliers(last_curve, 5, 0, plane='xy', show=0)
    plane_xy = last_curve_temp1[last_curve_temp1[:,-1]!=0, :-1]
    plane_xy=plane_xy[plane_xy[:, -1] != 0]
    x_data, y_data = plane_xy[:, 0], plane_xy[:, 1]
    params_opt, x_fit, y_fit = fit_curve_with_constraints(x_data, y_data ,model_line_function, initial_guess_line,show=0)

    last_curve_temp2 = knn_filter_outliers(last_curve, 3, 2, plane='xz', show=0)
    plane_xz = last_curve_temp2[last_curve_temp2[:, -1] != 0][:, [0, 2]]
    plane_xz=plane_xz[plane_xz[:, -1] != 0]
    x_data, z_data = plane_xz[:, 0], plane_xz[:, 1]
    params_opt, x_fit, z_fit = fit_curve_with_constraints(x_data, z_data ,model_curve_function, initial_guess_curve,show=0)

    p3d=np.vstack([x_fit,y_fit,z_fit])

    return p3d.T
bounce_indices=find_bounce(p3d)
bounce_indices = np.array(bounce_indices)
p3d_segment=np.split(p3d,bounce_indices)

p3d_1=p3d_segment[0]
p3d_2=p3d_segment[1]
p3d_3=p3d_segment[2]
p3d_4=p3d_segment[3]
p3d_5=p3d_segment[4]
p3d_6=p3d_segment[5]

service_indices=find_service(p3d_1)
service_trace,first_curve=np.split(p3d_1,[service_indices[0]])

service_trace=fit_service(service_trace)

first_curve = np.insert(first_curve, 0, service_trace[-1],axis=0)
first_curve = np.insert(first_curve, -1, p3d[bounce_indices[0]],axis=0)
first_trace =fit_first_curve(first_curve)

p3d_2=np.insert(p3d_2, 0, first_trace[-1],axis=0)
hit_indice_2=find_hit(p3d_2)
second_curve1,second_curve2=np.split(p3d_2,hit_indice_2)

second_trace1=fit_second_curve1(second_curve1)

second_curve2=np.insert(second_curve2, 0, second_trace1[-1],axis=0)
second_curve2=np.insert(second_curve2, -1, p3d[bounce_indices[1]],axis=0)
second_trace2=fit_second_curve2(second_curve2)
second_trace2 = second_trace2[::-1]

p3d_3=np.insert(p3d_3, 0, second_trace2[-1],axis=0)
hit_indice_3=find_hit(p3d_3)
third_curve1,third_curve2=np.split(p3d_3,hit_indice_3)

third_trace1=fit_third_curve1(third_curve1)
third_trace1=third_trace1[::-1]

third_curve2=np.insert(third_curve2, 0, third_curve1[-1],axis=0)
third_curve2=np.insert(third_curve2, -1, p3d[bounce_indices[2]],axis=0)
third_trace2=fit_third_curve2(third_curve2)

p3d_4=np.insert(p3d_4, 0, third_trace2[-1],axis=0)
hit_indice_4=find_hit(p3d_4)
fourth_curve1,fourth_curve2=np.split(p3d_4,hit_indice_4)

fourth_trace1=fit_fourth_curve1(fourth_curve1)

fourth_curve2=np.insert(fourth_curve2, 0, fourth_curve1[-1],axis=0)
fourth_curve2=np.insert(fourth_curve2, -1, p3d[bounce_indices[3]],axis=0)
fourth_trace2=fit_fourth_curve2(fourth_curve2)
fourth_trace2=fourth_trace2[::-1]

p3d_5=np.insert(p3d_5, 0, fourth_trace2[-1],axis=0)
hit_indice_5=find_hit(p3d_5)[0]
fifth_curve1,fifth_curve2=np.split(p3d_5,[hit_indice_5])

fifth_trace1=fit_fifth_curve1(fifth_curve1)
fifth_trace1=fifth_trace1[::-1]

fifth_curve2=np.insert(fifth_curve2, 0, fifth_trace1[-1],axis=0)
fifth_curve2=np.insert(fifth_curve2, -1, p3d[bounce_indices[4]],axis=0)
fifth_trace2=fit_fifth_curve2(fifth_curve2)

p3d_6=np.insert(p3d_6, 0, fifth_trace2[-1],axis=0)
last_trace=fit_last_curve(p3d_6)
hit_indices=find_hit(p3d)
bounce_indices=find_bounce(p3d)
p3d=np.vstack([service_trace,first_trace,second_trace1,second_trace2,third_trace1,third_trace2,fourth_trace1,fourth_trace2,fifth_trace1,fifth_trace2,last_trace])
indices = []
traces = [service_trace, first_trace, second_trace1, second_trace2, third_trace1, third_trace2, fourth_trace1, fourth_trace2, fifth_trace1, fifth_trace2]

for trace in traces:
    index = np.where((p3d == trace[-1]).all(axis=1))[0]
    indices.append(index)

green_indices = np.asarray(indices[:-1])
red_indices = np.asarray(indices[-1:])

x = p3d[:, 0]
y = p3d[:, 1]
z = p3d[:, 2]

def set_box_aspect(ax, aspect):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    xrange = max(xlim) - min(xlim)
    yrange = max(ylim) - min(ylim)
    zrange = max(zlim) - min(zlim)

    if aspect == 'equal':
        max_range = max(xrange, yrange, zrange)
        xlim = [np.mean(xlim) - max_range / 2, np.mean(xlim) + max_range / 2]
        ylim = [np.mean(ylim) - max_range / 2, np.mean(ylim) + max_range / 2]
        zlim = [np.mean(zlim) - max_range / 2, np.mean(zlim) + max_range / 2]
    elif aspect == 'auto':
        pass
    else:
        raise ValueError(f"Invalid value for aspect: {aspect}")

    ax.set_box_aspect([20, 10, 10])

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

court_points = np.load(r'./world_coordinates.npy', allow_pickle=True).item()
court_points = np.asarray(court_points['world_coordinates'])

point_groups = [
    [0, 4, 20, 16, 0],
    [1, 3, 19, 17, 1],
    [5, 7, 15, 13, 5],
    [6, 14],
    [8, 21, 25, 12]
]

for group in point_groups:
    court_x = court_points[group, 0]
    court_y = court_points[group, 1]
    court_z = court_points[group, 2].reshape((1, -1))
    ax.plot_wireframe(court_x, court_y, court_z, color='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_zlim(0, 5)
ax.set_xlim(-1, 24)
ax.set_ylim(-1, 11)

set_box_aspect(ax, 'equal')

def update(frame):
    ax.cla()
    for group in point_groups:
        court_x = court_points[group, 0]
        court_y = court_points[group, 1]
        court_z = court_points[group, 2].reshape((1, -1))
        ax.plot_wireframe(court_x, court_y, court_z, color='black')

    def is_near_index(index, indices, margin=1):
        for idx in indices:
            if abs(index - idx) <= margin:
                return True
        return False

    blue_points = []
    for i in range(frame):
        if not is_near_index(i, green_indices) and not is_near_index(i, red_indices):
            blue_points.append(i)

    ax.scatter(p3d[blue_points, 0], p3d[blue_points, 1], p3d[blue_points, 2], color='blue')

    for indice in green_indices:
        if indice <= frame:
            if not is_near_index(indice, red_indices):
                ax.scatter(p3d[indice, 0], p3d[indice, 1], p3d[indice, 2], color='green')

    for indice in red_indices:
        if indice <= frame:
            if not is_near_index(indice, green_indices):
                ax.scatter(p3d[indice, 0], p3d[indice, 1], p3d[indice, 2], color='red')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_zlim(0, 5)
    ax.set_xlim(-1, 24)
    ax.set_ylim(-1, 11)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)]
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10))
    plt.legend(handles, ['in/hit', 'out'], loc='upper right', frameon=False)

update(len(p3d))
plt.show()


