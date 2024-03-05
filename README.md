# The project is based on the Robot Vision course at **Southeast University(SEU)** in 2023. #

## **Introduction**
The goal of the project is to develop a tennis Hawkeye system based on videos captured by four cameras on the court to determine whether the ball is out of bounds.

![tennis_animation.gif-3929.5kB][1]
## **Challenges**
### 1. **Camera Calibration**  
We selected 21 representative points as reference,as shown in the following picture:
![A1.png-1073.1kB][2]
Camera calibration is the foundation of 3D reconstruction. We need to obtain both the intrinsic and extrinsic parameters of the camera.The following are several methods and comparison of their effects.  

- **Checkerboard calibration and pnp solution to camera matrix.**  
First, we use the chessboard taken by the camera to solve the internal parameter matrix:
![](https://static.zybuluo.com/zgh456/scry2y8yfft3eihmwuqef5te/in.png)
then we use the function `cv2.solvePnP` to get the **Rotation matrix** and the **Translation vector**.But the result is not good:
![pnp1.png-75kB][3]
![pnp2.png-44.7kB][4]
I wonder why this is happening. Let's delve into its underlying causes
- **SVD decomposition for solving camera matrix.**  
The foundation of obtaining the camera matrix lies in solving the least squares solution from three-dimensional to two-dimensional mapping.In summary, we need to obtain the matrix with 12 parameters **P**.  
So we can create a zero matrix M with a size of 3*points rows and 12+points columns. Then, for each three-dimensional point, assign its coordinates to different rows and columns of matrix M. Finally, assign the negative values of the two-dimensional point coordinates to specific columns of matrix M. Matrix M is used for the subsequent calculation of the camera projection matrix.
Let's see how it turns out:
![svd.png-66.1kB][5]
This result looks much better. However, there is still an issue. Different methods of matrix normalization yield different solutions. We need to try multiple normalization techniques to achieve a better result. Even though the current image result looks fine, there are still some deviations numerically. Can we further improve it?
- **Neural networks for solving camera matrix.** 
We know that the purpose of training a neural network is essentially to find the weights for each parameter. Therefore, we can construct a single-layer neural network to find our least squares solution, while ensuring to deactivate the bias term during training.
![court_linear_regression.png-59.9kB][6]
This result achieves the current best performance, with only very small numerical errors.
<span style="color:red">**We cannot simply select points on a two-dimensional plane, as the least squares solution does not impose constraints on the three direction vectors of the extrinsic parameters to be mutually orthogonal. Therefore, the points we select must ensure that they lie within a three-dimensional plane.**</span>

### 2. **Ball Tracking**  
- **Gaussian Mixture Background Subtractor + SIFT**  
Firstly, we extract several images of tennis balls from the photo and adjust their size, brightness, and contrast to ensure diversity of descriptors. Then, we place them in the "foreground" folder. Next, we place the pure background images in the "background" folder. These two folders are used for extracting descriptors of foreground and background using SIFT.
Secondly, we use a background subtractor to obtain the foreground, apply median filtering for refinement, and finally utilize SIFT to extract descriptors from the images and find matching pixel points corresponding to tennis balls.At the same time, we incorporate the descriptors of the newly detected tennis balls to ensure the continuity of detection.
However, due to the inevitable occurrence of mistakenly recognizing the background as tennis balls and incorporating them into the descriptors, the detection algorithm may not perform with sufficient precision, but it can still be used.
![sift.png-543kB][7]
**I also added a Kalman filter in the code, but did not call it because the effect was not satisfactory.**
- **YOLO**  
We used a model trained on 9000 images of tennis matches from the Roboflow website to detect tennis balls, and the effectiveness has significantly improved.
![yolo.png-265.8kB][8]

Due to the limited video resolution, we cannot achieve accurate recognition for every frame. Therefore, we preserve the two-dimensional point arrays containing noise points and filter them out during reconstruction.
### 3. **Reconstruction**  
We first need to align the time series of the videos, and then use `cv2.triangulatePoints` to calculate three-dimensional coordinates. However, due to the presence of noise points, we cannot reconstruct every point accurately, and these noise points will also affect our subsequent curve fitting. Therefore, the challenge here lies in how to remove noise points and how to fit curves accurately.
**Original Image:**
![real.png-106.8kB][9]
We first use the mean and variance of each point's neighborhood to determine whether the point is an outlier.
**Image after removing outliers:**
![original.png-95.3kB][10]
The next step is what's been puzzling me for a while now because the noise is sparse, clumpy, and clustered around the trajectory of the ball, making it difficult to filter out.  
But after reconstruction, we have obtained a new perspective, which means we can observe the trajectory of the ball from any angle. So, we can view the field from above and observe the trajectory from the side. We have shifted the problem from filtering three-dimensional points to filtering two-dimensional points and use the K-nearest neighbors (KNN) algorithm to compute the distances and indices to the nearest neighbors of each point in order to remove outliers.
**The image after KNN filtering:**
![look_down.png-24.1kB][11]
![1.png-93.6kB][12]
Now we can fit each segment of the curve.
The next issue is the discontinuity between each fitted curve segment. So, we identify each point of impact and ground contact separately, assign them individual losses, and use `scipy.optimize.minimize` for curve fitting.
**Bounces in image(red points):**
![b.png-87.4kB][13]
**Loss function:**
```
def loss_func(params, x, y, start_point, end_point):
    predicted_y = model_func(params, x)
    start_loss = (predicted_y[0] - start_point[1]) ** 2
    end_loss = (predicted_y[-1] - end_point[1]) ** 2
    data_loss = np.mean((predicted_y - y) ** 2)
    return start_loss + end_loss + data_loss
```

**The filtering can also be done using epipolar geometry:**  
Utilize the `cv2.computeCorrespondEpilines` function to compute the epipolar lines corresponding to each point in the first set of points on the second image. Then, calculate the distance from each point in the second set of points to the corresponding epipolar line. Check if the distance of the first point in the first set to its corresponding epipolar line is less than 100. If so, return True, indicating that these two points match; otherwise, return False, indicating that they do not match. However, the effectiveness of this method may be limited.  
**Finally, we obtain our fitted curve:**
![last.png-65.3kB][14]
## **Usage**  
### 1. **Establish world coordinates.**  
Enter the real-world coordinates in the world_coordinates.py file.You can arbitrarily choose the origin.
![1R_forPnP2.jpg-93.2kB][15]
### 2. **Select points**  
According to the world coordinates you've established, use select_points.py to **sequentially** select the corresponding points.
![A1.png-1073.1kB][16]
### 3. **Compute camera matrix**  
Runn the least_square_solution_net.py file to compute the camera matrix P for each camera.Then your camera matrices will be stored in a .npy file.
### 4. **Get points**  
Run sift_tracker.py to obtain the two-dimensional coordinates of the tennis ball.
![URQ3GR{D5{S9~@(R{J9ED25.png-6.6kB][17]
### 5. **Reconstruction**  
Finally, run RECONSTRUCTION.py to yield the results of the three-dimensional reconstruction.
![a.png-107.1kB][18]
## Contributors
- [Contributor1](https://github.com/GehaoZhang6)




  [1]: https://static.zybuluo.com/zgh456/ngxwl40fw0jd8yeetd1r6jad/tennis_animation.gif
  [2]: https://static.zybuluo.com/zgh456/erortd875za5k897rk1zbz0p/A1.png
  [3]: https://static.zybuluo.com/zgh456/6r7qsk436edztlsasj7h0yw5/pnp1.png
  [4]: https://static.zybuluo.com/zgh456/6krgfwfeopr9s4bfd45jimqr/pnp2.png
  [5]: https://static.zybuluo.com/zgh456/kxrs5mngao6uadr6ciy6s7xr/svd.png
  [6]: https://static.zybuluo.com/zgh456/dvsn8tz96bncpui5vnvhwmj7/court_linear_regression.png
  [7]: https://static.zybuluo.com/zgh456/z8b08celo9nywtwavhql925f/sift.png
  [8]: https://static.zybuluo.com/zgh456/8wv197p364zu1all4lvo9t9m/yolo.png
  [9]: https://static.zybuluo.com/zgh456/0tlyf1qmzsgm0070flbeq04k/real.png
  [10]: https://static.zybuluo.com/zgh456/rotws4uarou8vh2in8onljf2/original.png
  [11]: https://static.zybuluo.com/zgh456/2qz947x4hleeyovujsdyvlhu/look_down.png
  [12]: https://static.zybuluo.com/zgh456/36wdv50ga95aq2eeqmszopnc/1.png
  [13]: https://static.zybuluo.com/zgh456/2zgu3d1hebt3dyhhgktfnel9/b.png
  [14]: https://static.zybuluo.com/zgh456/rwocinuo08be6rfa63kr72tp/last.png
  [15]: https://static.zybuluo.com/zgh456/1253kyyw0q78cr6h0vek75bd/1R_forPnP2.jpg
  [16]: https://static.zybuluo.com/zgh456/mbbnpiesfw5joa4l7ke2tkpv/A1.png
  [17]: https://static.zybuluo.com/zgh456/a2c9gv39hdi0e975tby7luxc/URQ3GR%7BD5%7BS9~@%28R%7BJ9ED25.png
  [18]: https://static.zybuluo.com/zgh456/xi61haget7becfre110kvj4k/a.png
