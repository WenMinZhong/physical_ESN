import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ESN
import numpy as np
import torch
import time
device=torch.device('cuda:0')
lorenz_data = np.load('数据集/lorenz_dataset.npz')
curve_0, curve_1, curve_2 = lorenz_data['arr_0'], lorenz_data['arr_1'], lorenz_data['arr_2']
inSize = 1
outSize = 1
resSize = 100
def linear_regression(x_train, y_train, x_test, y_test,alpha):
    Echo = ESN.ESN(inSize, outSize, resSize, alpha)
    RA_train=Echo.reservoir(x_train).to(torch.float64)
    y_train,y_test=torch.tensor(y_train,device=device),torch.tensor(y_test,device=device)
    w = torch.matmul(torch.linalg.pinv(torch.matmul(RA_train.transpose(0,1), RA_train)), torch.matmul(RA_train.transpose(0,1), y_train))
    RA_test = Echo.reservoir(x_test).to(torch.float64)
    import time
    current_time = time.time()
    local_time = time.localtime(current_time)
    path='btype'
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    if alpha==1:
        torch.save(RA_test,path+'/'+str(local_time.tm_hour)+str(local_time.tm_min)+str(local_time.tm_sec)+'.pt')
    if alpha==0.8:
        torch.save(RA_test,path+'/'+str(local_time.tm_hour)+str(local_time.tm_min)+str(local_time.tm_sec)+'.pt')
    if alpha==0.6:
        torch.save(RA_test,path+'/'+str(local_time.tm_hour)+str(local_time.tm_min)+str(local_time.tm_sec)+'.pt')

    predict = torch.matmul(RA_test, w)
    RMSE = torch.sqrt(((predict[100:-1] - y_test[100:-1]) ** 2).mean())
    range_of_o = y_test[100:-1].max() - y_test[100:-1].min()
    nrmse = RMSE / range_of_o
    return predict, nrmse,w


def train(train_input, train_target, test_input, test_target):
    predict, RMSE,w = linear_regression(train_input, train_target, test_input, test_target)
    return predict, train_target, RMSE, w

def plot(x,y,z):
    plt.subplot(3, 2, 1)
    plt.plot(x[-100:-1], color='black')
    plt.xlabel('Step(#)')
    plt.ylabel('a.u.')
    plt.title('x')
    plt.subplot(3, 2, 3)
    plt.plot(y[-100:-1], color='red')
    plt.title('y')
    plt.xlabel('Step(#)')
    plt.ylabel('a.u.')
    plt.subplot(3, 2, 5)
    plt.title('z')
    plt.xlabel('Step(#)')
    plt.ylabel('a.u.')
    plt.plot(z[-100:-1], color='blue')
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.plot(x[-100:-1], y[-100:-1], z[-100:-1], color='green')
    ax.grid(False)
    ax.set_facecolor('none')
    plt.tight_layout()
    plt.show()

def main(alpha):
    x_train_input, x_train_target, x_test_input, x_test_target=curve_0[0:4998],curve_0[2:5000],curve_0[5000:9998],curve_0[5002:10000]
    y_train_input, y_train_target, y_test_input, y_test_target=curve_1[0:4998],curve_1[2:5000],curve_1[5000:9998],curve_1[5002:10000]
    z_train_input, z_train_target, z_test_input, z_test_target=curve_2[0:4998],curve_2[2:5000],curve_2[5000:9998],curve_2[5002:10000]
    x, x_nrmse, x_w=linear_regression(x_train_input, x_train_target, x_test_input, x_test_target,alpha)
    y, y_nrmse, y_w = linear_regression(y_train_input, y_train_target, y_test_input, y_test_target,alpha)
    z, z_nrmse, z_w = linear_regression(z_train_input, z_train_target, z_test_input, z_test_target,alpha)
    x, y, z = x.detach().cpu().numpy(), y.detach().cpu().numpy(), z.detach().cpu().numpy()
    x_nrmse,y_nrmse,z_nrmse=x_nrmse.detach().cpu().numpy(),y_nrmse.detach().cpu().numpy(),z_nrmse.detach().cpu().numpy()
    # result=np.array([x, y, z])
    # nrmse=np.array([x_nrmse,y_nrmse,z_nrmse])
    return np.array([x, y, z]), np.array([x_nrmse,y_nrmse,z_nrmse])

if __name__ == '__main__':
    pre_result = []
    nrmse_result = []
    alpha_list=[0.8]
    for i in alpha_list:
        alpha=i
        predict_value, nrmse = main(alpha)
        pre_result.append(predict_value)
        nrmse_result.append(nrmse)
    pre_result = np.array(pre_result)
    nrmse_result=np.array(nrmse_result)
    print(pre_result.shape)
    import pickle
    dic={'100predict':pre_result,'100nrmse':nrmse_result}
    with open('b_data-100','wb') as f:
        pickle.dump(dic,f)

    # array = np.array([x, y, z])
    x,y,z=pre_result[0][0],pre_result[0][1],pre_result[0][0]
    plot(x, y, z)
    print(nrmse_result)





