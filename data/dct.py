import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# N*N离散余弦变换 (DCT) 矩阵
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

# 2D频率选择掩码（Mask）的函数。它主要用于在 离散余弦变换 (DCT) 域中筛选特定的频率分量。
def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

# Sigmoid 激活函数 ，将其输出范围从默认的(0,1)映射到了(-1,1)
def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

#频率域特征提取器。它专门用于对 2D 离散余弦变换（DCT）后的频谱图进行筛选和加权。
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

#将一张大图拆解，并在频率域中筛选出“信息量最大”（最复杂）和“信息量最小”（最平滑）的局部图像块。
class DCT_base_Rec_Module(nn.Module):
    """_summary_

    Args:
        x: [C, H, W] -> [C*level, output, output]
    """
    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_fliter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)
        
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)
        
        #patch块数量= [（输入尺寸256-窗口大小32）/步长16] +1，使用向下取整的方式处理边缘
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), 
            kernel_size=(window_size, window_size), 
            stride=window_size
        )
        
        lm, mh = 2.82, 2
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])
        
        
    def forward(self, x):
        
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, W, H = x.shape
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)  
        

        #将图像划分为patch，并将每个独立patch通过DCT转换到频率域
        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size) 
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        
        #使用每个level滤波器进行滤波
        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            #转换回空间域，并进行拼接
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)
        
        #计算每个patch的grade得分
        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[1,2,3])
            grade += w * _x            
            w *= k
        
        #选取两个最低和最高grade的patch
        _, idx = torch.sort(grade)
        max_idx = torch.flip(idx, dims=[0])[:N]
        maxmax_idx = max_idx[0]
        if len(max_idx) == 1:
            maxmax_idx1 = max_idx[0]
        else:
            maxmax_idx1 = max_idx[1]

        min_idx = idx[:N]
        minmin_idx = idx[0]
        if len(min_idx) == 1:
            minmin_idx1 = idx[0]
        else:
            minmin_idx1 = idx[1]

        #从level_x_unfold中选取两个最低和最高grade的patch
        #得分最高的图块（全图中纹理最复杂的块，通常包含核心物体边缘或密集细节）。
        #得分最低的图块（全图中最为平滑的块，如天空、白墙或背景）。
        x_minmin = torch.index_select(level_x_unfold, 0, minmin_idx)
        x_maxmax = torch.index_select(level_x_unfold, 0, maxmax_idx)
        x_minmin1 = torch.index_select(level_x_unfold, 0, minmin_idx1)
        x_maxmax1 = torch.index_select(level_x_unfold, 0, maxmax_idx1)

        #将筛选出的 单个图像块（Patch）的多维张量转换成 nn.Fold 能够接收的 列向量格式
        x_minmin = x_minmin.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_maxmax = x_maxmax.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_minmin1 = x_minmin1.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_maxmax1 = x_maxmax1.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)

        #将列向量格式的patch重新拼接成图像块32*32的格式
        x_minmin = self.fold0(x_minmin)
        x_maxmax = self.fold0(x_maxmax)
        x_minmin1 = self.fold0(x_minmin1)
        x_maxmax1 = self.fold0(x_maxmax1)

       
        return x_minmin, x_maxmax, x_minmin1, x_maxmax1


        


