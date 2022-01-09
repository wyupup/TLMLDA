# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:35:10 2020

@author: ZHUANG
"""
import torch

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])  
    total = torch.cat([source, target], dim=0) 
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list] 
    return sum(kernel_val) 
                                                                                                         
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):    
    batch_size = source.size()[0] 
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = torch.Tensor([0]).cuda()   
    if torch.sum(torch.isnan(sum(kernels))):  
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]
    
    loss = torch.mean(SS + TT - 2 * ST)
    return loss