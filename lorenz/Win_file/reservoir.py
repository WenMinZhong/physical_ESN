import numpy as np
import torch
device = torch.device('cuda:0')
def nVg_18_p(x):
    return -1.80516576e-12 * x ** 3 + -3.38090834e-10 * x ** 2 + -2.65468792e-09 * x + -8.04465828e-09
def nVg_24_p(x):
    return 2.02728121e-11 * x ** 3 + 5.96515544e-10 * x ** 2 + 6.55555741e-09 * x + 8.77849099e-09
def nVg_30_p(x):
    return 7.36173225e-12 * x ** 3 + 2.22140057e-10 * x ** 2 + 3.83185081e-09 * x + 4.85986551e-09
def nVg_18_n(x):
    return 9.85756414e-08*(1-torch.exp(-1*5.21588049e-01*x))
def nVg_24_n(x):
    return 3.40832433e-07*(1-torch.exp(-1*3.27713687e-01*x))
def nVg_30_n(x):
    return 9.38417464e-07*(1-torch.exp(-1*1.79543199e-01*x))
def nconect18(x):
    return (1- torch.sign(x))*0.5*nVg_18_p(x)+(1+torch.sign(x))*0.5*nVg_18_n(x)
def nconect24(x):
    return (1- torch.sign(x))*0.5*nVg_24_p(x)+(1+torch.sign(x))*0.5*nVg_24_n(x)
def nconect30(x):
    return (1- torch.sign(x))*0.5*nVg_30_p(x)+(1+torch.sign(x))*0.5*nVg_30_n(x)
def ndevice1(x):
    return nconect18(x)*10e6
def ndevice2(x):
    return nconect24(x)*10e6
def ndevice3(x):
    return nconect30(x)*10e6

##ambipolar
def bVg_18_p(x):
    return -1.61312101e-07*(1-torch.exp(-1*-1.44528974e-01*x))
def bVg_24_p(x):
    return -2.54148484e-07*(1-torch.exp(-1*-1.13508001e-01*x))
def bVg_30_p(x):
    return -3.75952449e-07*(1-torch.exp(-1*-8.98657703e-02*x))
def bVg_18_n(x):
    return 1.71444530e-07*(1-torch.exp(-1*2.00662093e-01*x))
def bVg_24_n(x):
    return 2.48514331e-07*(1-torch.exp(-1*1.78401776e-01*x))
def bVg_30_n(x):
    return 3.47736709e-07*(1-torch.exp(-1*1.57677791e-01*x))
def bconect18(x):
    return (1- torch.sign(x))*0.5*bVg_18_p(x)+(1+torch.sign(x))*0.1*bVg_18_n(x)
def bconect24(x):
    return (1- torch.sign(x))*0.5*bVg_24_p(x)+(1+torch.sign(x))*0.1*bVg_24_n(x)
def bconect30(x):
    return (1- torch.sign(x))*0.5*bVg_30_p(x)+(1+torch.sign(x))*0.1*bVg_30_n(x)
def bdevice1(x):
    return bconect18(x)*10**6
def bdevice2(x):
    return bconect24(x)*10**6
def bdevice3(x):
    return bconect30(x)*10**6


##p
def pVg_18_p(x):
    return 1.80400290e-11 * x ** 3 + 9.98677095e-10 * x ** 2 + 2.19445016e-08 * x + 8.00742911e-10
def pVg_24_p(x):
    return 1.80306470e-11 * x ** 3 + 1.12413121e-09 * x ** 2 + 2.74864579e-08 * x + 7.29213541e-09
def pVg_30_p(x):
    return 1.37234864e-11 * x ** 3 + 1.03177352e-09 * x ** 2 + 3.03896563e-08 * x + 9.58431663e-09
def pVg_18_n(x):
    return 1.83418198e-11 * x ** 3 + -4.81767569e-10 * x ** 2 + 3.92371825e-09 * x + 1.20325320e-08
def pVg_24_n(x):
    return 1.83418198e-11 * x ** 3 + -4.81767569e-10 * x ** 2 + 3.92371825e-09 * x + 1.20325320e-08
def pVg_30_n(x):
    return 1.83418198e-11 * x ** 3 + -4.81767569e-10 * x ** 2 + 3.92371825e-09 * x + 1.20325320e-08
def pconect18(x):
    return (1- torch.sign(x))*0.5*pVg_18_p(x)+(1+torch.sign(x))*0.5*pVg_18_n(x)
def pconect24(x):
    return (1-torch.sign(x))*0.5*pVg_24_p(x)+(1+torch.sign(x))*0.5*pVg_24_n(x)
def pconect30(x):
    return (1- torch.sign(x))*0.5*pVg_30_p(x)+(1+torch.sign(x))*0.5*pVg_30_n(x)
def pdevice1(x):
    return pconect18(x)*10e6
def pdevice2(x):
    return pconect24(x)*10e6
def pdevice3(x):
    return pconect30(x)*10e6

def boperation(input, W):
    W.to(device)
    Result = torch.tile(input,(len(W),1)).transpose(0,1).cuda()


    Result = torch.where(W == 0, 0, Result)
    Result = torch.where(W == 1, bdevice1(Result), Result)
    Result= torch.where(W == 2, bdevice2(Result), Result)
    Result = torch.where(W == 3, bdevice3(Result), Result)
    Result = torch.sum(Result, dim=0)
    # print('result',np.max(Result),np.min(Result))

    return Result
def noperation(input, W):
    Result = torch.tile(input,(len(W),1)).transpose(0,1)
    Result = torch.where(W == 0, 0, Result)
    Result = torch.where(W == 1, ndevice1(Result), Result)
    Result= torch.where(W == 2, ndevice2(Result), Result)
    Result = torch.where(W == 3, ndevice3(Result), Result)
    Result = torch.sum(Result, dim=0)
    # print('result',np.max(Result),np.min(Result))
    return Result

def poperation(input, W):
    Result = torch.tile(input,(len(W),1)).transpose(0,1)
    Result = torch.where(W == 0, 0, Result)
    Result = torch.where(W == 1, pdevice1(Result), Result)
    Result= torch.where(W == 2, pdevice2(Result), Result)
    Result = torch.where(W == 3, pdevice3(Result), Result)
    Result = torch.sum(Result, dim=0)
    # print('result',np.max(Result),np.min(Result))
    return Result
# Resize=16
# W = torch.randint(low=0, high=4, size=(16, 16))
#
# input = np.random.random(16)-0.5
# print(input)
# print(operation(input,W,Resize))