import torch
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib

plt.rcParams['font.size'] = 10  # 设置默认字体大小为12
plt.rcParams['font.family'] = 'Arial'  # 设置默认字体类型为Arial
print(torch.version.cuda)
device = torch.device('cuda:0')
data = np.load('数据集/mnist.npz')
x_train, y_train, x_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
nRA_train = torch.load('ntype_RA/RA_train0.0.pt')
nRA_test = torch.load('ntype_RA/RA_test0.0.pt')
pRA_train = torch.load('ptype_RA/RA_train0.0.pt')
pRA_test = torch.load('ptype_RA/RA_test0.0.pt')
bRA_train = torch.load('ambipolar_RA/RA_train0.0.pt')
bRA_test = torch.load('ambipolar_RA/RA_test0.0.pt')
RA_train = torch.load('RawESNRA/fashion_RA_train0.pt')
RA_test = torch.load('RawESNRA/fashion_RA_test0.pt')
y_train = torch.tensor(y_train, device=device, dtype=torch.float64)
y_test = torch.tensor(y_test, device=device, dtype=torch.float64)


def prediction(RA_train, RA_test, y_train, y_test):
    RA_train = RA_train.to(torch.float64)
    RA_test = RA_test.to(torch.float64)
    w = torch.matmul(torch.linalg.pinv(torch.matmul(RA_train.transpose(0,1),RA_train)),torch.matmul(RA_train.transpose(0,1),y_train))
    predict = torch.matmul(RA_test, w)
    y_test = torch.argmax(y_test, dim=1)
    predict = torch.argmax(predict, dim=1)
    return predict.detach().cpu().numpy(), y_test.detach().cpu().numpy()

def evaluate(predict, y_test):
    acc = predict==y_test
    acc = predict.to(torch.int32)
    acc = torch.sum(acc) / len(acc)
    return acc

# def confusion_matrix(predictions, targets):
#     num_classes = predictions.max() + 1
#     cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
#     for t, p in zip(targets.view(-1), predictions.view(-1)):
#         cm[t.long(), p.long()] += 1
#     return cm.detach().cpu().numpy()



nconfusion_matrix = torch.load('ntype_RA/confusion0.0.pt')
bconfusion_matrix = torch.load('ambipolar_RA/confusion0.0.pt')
pconfusion_matrix = torch.load('ptype_RA/confusion0.0.pt')
confusion_matrix = torch.load('RawESNRA/confusion.pt')



fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 3)
colors = ['#7F8080', '#D1ABCF', "#8BCCB5"]
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name, colors)

name =['0','1',  '2',  '3',  '4', '5', '6', '7', '8', '9']
ax0 = fig.add_subplot(gs[0])
cax0 = ax0.imshow(nconfusion_matrix, cmap='jet')
fmt = 'd'
acc=np.trace(nconfusion_matrix)/np.sum(nconfusion_matrix)*100
thresh = nconfusion_matrix.max() / 2.
# cbar0 = fig.colorbar(cax0, ax=ax0, orientation='vertical', fraction=0.046, pad=0.04)
ax0.set_xticks((range(10)), (name))
ax0.set_yticks((range(10)), (name))
ax0.set_xlabel('Predict', fontsize=12, fontfamily='Arial')
ax0.set_ylabel('True', fontsize=12, fontfamily='Arial')
ax0.set_title('ntype acc='+"{:.2f}".format(acc)+'%',fontsize=10)
# cbar0.ax.tick_params(labelsize=8)

ax1 = fig.add_subplot(gs[1])
cax1 = ax1.imshow(bconfusion_matrix, cmap='jet')
fmt = 'd'
thresh = bconfusion_matrix.max() / 2.
ax1.set_xticks((range(10)), (name))
ax1.set_yticks((range(10)), (name))
ax1.set_xlabel('Predict', fontsize=12, fontfamily='Arial')
ax1.set_ylabel('True', fontsize=12, fontfamily='Arial')
acc=np.trace(bconfusion_matrix)/np.sum(bconfusion_matrix)*100
ax1.set_title('ambipolar acc='+"{:.2f}".format(acc)+'%',fontsize=10)
# cbar1.ax.tick_params(labelsize=8)

ax2 = fig.add_subplot(gs[2])
cax2 = ax2.imshow(pconfusion_matrix, cmap='jet')
fmt = 'd'
thresh = pconfusion_matrix.max() / 2.
ax2.set_xticks((range(10)), (name))
ax2.set_yticks((range(10)), (name))
ax2.set_xlabel('Predict', fontsize=12, fontfamily='Arial')
ax2.set_ylabel('True', fontsize=12, fontfamily='Arial')
acc=np.trace(pconfusion_matrix)/np.sum(pconfusion_matrix)*100
ax2.set_title('ptype acc='+"{:.2f}".format(acc)+'%',fontsize=10)
plt.tight_layout()
# plt.rcParams['svg.fonttype'] = 'none'
plt.savefig('Fconfusion.svg',transparent=True,dpi=600,bbox_inches='tight')
plt.show()
