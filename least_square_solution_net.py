import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def pflat(x):
    y = np.copy(x)
    for i in range(x.shape[1]):
        y[:, i] = y[:, i] / y[x.shape[0] - 1, i]
    return y

class LinearSolver(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearSolver, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x = x.to(torch.float32)
        return self.linear(x)


def is_zero(p: np.ndarray) -> bool:

    if p[0] == 0 and p[1] == 0:
        return True
    else:
        return False


def calc_P(p3d: np.ndarray, p2d: np.ndarray) -> np.ndarray:

    npoints = p2d.shape[1]

    mean = np.mean(p2d, 1)
    std = np.std(p2d, axis=1)

    N = np.array([
        [1 / std[0], 0, -mean[0] / std[0]],
        [0, 1 / std[1], -mean[1] / std[1]],
        [0, 0, 1],
    ])

    p2dnorm = np.matmul(N, p2d)
    M = np.zeros([3 * npoints, 12 + npoints])
    for i in range(npoints):
        M[3 * i, 0:4] = p3d[:, i]
        M[3 * i + 1, 4:8] = p3d[:, i]
        M[3 * i + 2, 8:12] = p3d[:, i]
        M[3 * i: 3 * i + 3, 12 + i] = -p2dnorm[:, i]

    M = torch.tensor(M, dtype=torch.float32)
    b = torch.zeros((3 * npoints, 1), dtype=torch.float32)
    print('M',M)
    print(M)

    input_dim = M.shape[1]
    output_dim = b.shape[1]
    model = LinearSolver(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.MSELoss()

    num_epochs = 10000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(M)
        # 计算损失
        loss = criterion(outputs, b)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # 使用训练后的模型进行预测
    with torch.no_grad():
        predicted = model(M)
        print(predicted)
    weights = []
    for name, param in model.named_parameters():
        print(name, param.data)
        weights.append(param.data)
    v = [weight.squeeze().tolist() for weight in weights[:12]]
    print('X',v)
    P = np.reshape(v[0][0:12], [3, 4])
    testsign = np.matmul(P, p3d[:, 1])
    if testsign[2] < 0:
        P = -P

    P = np.matmul(np.linalg.inv(N), P)
    return P


objpoints=np.load(r'./world_coordinates.npy',allow_pickle=True).item()
objpoints=objpoints['world_coordinates']
objpoints = np.asarray(objpoints).astype(np.float32)
objpoints=objpoints[:][:]
objpoints=objpoints.T
objpoints1=np.vstack((objpoints,np.ones((1, objpoints.shape[1]))))
objpoints=np.vstack((objpoints,np.ones((1, objpoints.shape[1])))).T
imgpoints_L=np.load(r'./camera_params/camera_coordinate_2L.npy',allow_pickle=True)
imgpoints_L = np.asarray(imgpoints_L).astype(np.float32)
imgpoints_L=imgpoints_L[:][:]
imgpoints_L=imgpoints_L.T
imgpoints1=np.vstack((imgpoints_L,np.ones((1, imgpoints_L.shape[1]))))
imgpoints=np.vstack((imgpoints_L,np.ones((1, imgpoints_L.shape[1])))).T


P=calc_P(objpoints1, imgpoints1)

print(P.shape)

for i in range(objpoints.shape[0]):
    x=P@(objpoints[i,:])
    x=x[:]/x[-1]
    print(x)
    print(imgpoints[i,:])

extrinsic_camera_params = {
    "2L": P
}
# np.save("./camera_params/2L_to_world.npy", extrinsic_camera_params)


