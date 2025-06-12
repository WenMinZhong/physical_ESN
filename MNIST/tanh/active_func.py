import numpy as np
import torch
def bVg_30_p(x):
    return -3.75952449e-07*(1-torch.exp(-1*-8.98657703e-02*x))
def bVg_30_n(x):
    return 3.47736709e-07*(1-torch.exp(-1*1.57677791e-01*x))
def bconect30(x):
    return (1-torch.sign(x))*0.5*bVg_30_p(x)+(1+torch.sign(x))*0.35*bVg_30_n(x)
def btanh(x):
    return 1*bconect30(x)*10**6






def nVg_30_p(x):
    return 7.36173225e-12 * x ** 3 + 2.22140057e-10 * x ** 2 + 3.83185081e-09 * x + 4.85986551e-09
def nVg_30_n(x):
    return 9.38417464e-07*(1-torch.exp(-1*1.79543199e-01*x))
def nconect30(x):
    return (1- torch.sign(x))*0.5*nVg_30_p(x)+(1+torch.sign(x))*0.5*nVg_30_n(x)
def ntanh(x):
    return nconect30(x)*10**6



def pVg_30_p(x):
    return 1.37234864e-11 * x ** 3 + 1.03177352e-09 * x ** 2 + 3.03896563e-08 * x + 9.58431663e-09
def pVg_30_n(x):
    return 1.83418198e-11 * x ** 3 + -4.81767569e-10 * x ** 2 + 3.92371825e-09 * x + 1.20325320e-08
def pconect30(x):
    return (1- torch.sign(x))*0.5*pVg_30_p(x)+(1+torch.sign(x))*0.5*pVg_30_n(x)
def ptanh(x):
    return pconect30(x)*10**6