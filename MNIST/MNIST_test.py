import matplotlib.pyplot as plt
import numpy as np
import ESN
import torch
import ESN_raw
from sklearn.metrics import confusion_matrix
inSize = 784
outSize = 10
alpha = 1
resSize = 1000
device = torch.device('cuda:0')
data = np.load('数据集/fashion_dataset.npz')

x_train, y_train, x_test, y_test = data['arr_0'].reshape(60000,-1), data['arr_1'], data['arr_2'].reshape(10000,-1), data['arr_3']


# x_train = x_train/255
# x_test = x_test/255


y_train=np.eye(10)[y_train]
y_test=np.eye(10)[y_test]
print(y_train.shape)


for i in range(1):
    if i<=10:
        # noise_x_train = np.random.rand(x_train.shape[0],x_train.shape[1])
        # noise_x_train[np.random.rand(x_train.shape[0],x_train.shape[1])>=i*0.2]=0
        # noise_x_test = np.random.rand(x_test.shape[0],x_test.shape[1])
        # noise_x_test[np.random.rand(x_test.shape[0],x_test.shape[1]) >= i * 0.2] = 0
        # x_train = x_train+noise_x_train
        # x_test = x_test+noise_x_test
        Echo = ESN.ESN(inSize, outSize, resSize, alpha)
        RA_Train = Echo.reservoir(x_train).to(torch.float64)
        if torch.isnan(RA_Train).any():
            print("The matrix contains NaN values.")
        else:
            print("The matrix does not contain NaN values.")
        torch.save(RA_Train, 'ptype_RA/fashion_RA_train'+"{:.2f}".format(resSize)+'.pt')
        RA_Test = Echo.reservoir(x_test).to(torch.float64)
        torch.save(RA_Test, 'ptype_RA/fashion_RA_test'+"{:.2f}".format(resSize)+'.pt')
        y_train = torch.tensor(y_train,dtype=torch.float64).to(device)
        y_test = torch.tensor(y_test,dtype=torch.float64).to(device)
        RA_Test = torch.tensor(RA_Test,dtype=torch.float64)
        RA_Train = torch.tensor(RA_Train,dtype=torch.float64)
        w = torch.matmul(torch.linalg.pinv(torch.matmul(RA_Train.transpose(0,1),RA_Train)),torch.matmul(RA_Train.transpose(0,1),y_train))
        predict = torch.matmul(RA_Test,w)
        confusion=confusion_matrix(torch.argmax(y_test,dim=1).detach().cpu().numpy(),torch.argmax(predict,dim=1).detach().cpu().numpy())
        predict = torch.argmax(predict,dim=1)==torch.argmax(y_test,dim=1)
        torch.save(confusion,'ptype_RA/fashion'+"{:.2f}".format(resSize)+'.pt')
        acc = predict.to(torch.int32)
        acc = torch.sum(acc)/len(acc)
        # plt.imshow(confusion)
        # plt.show()
        print(acc)



