import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np
import torch.nn.functional as F

#将输入图像从空间域（像素）转换到残差域（噪声）。将输入图像经过 30 个 Spatial Rich Model (SRM) 滤波器进行卷积计算，并返回提取后的高频特征图
#冻结权重 (requires_grad=False）。这些 SRM 滤波器使用的是专家经验设计的固定权重，不参与神经网络的训练（反向传播不会更新它们）
# 过滤掉平滑区域：图像中大面积颜色相近的区域（如蓝天、白墙）在输出中会接近于 0（黑色）。
# 增强突变区域：图像中的边缘、物体轮廓、以及肉眼难以察觉的像素级“噪声”会被放大。
# 多维度分析：返回的 30 个通道分别代表了不同的噪声模式（例如：水平方向残差、垂直方向残差、二阶拉普拉斯残差等）。
class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
    hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
   

    self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight


  def forward(self, input):

    output = self.hpf(input)

    return output

# ==========================================
# 核心修改部分 1: 定义颜色转换与约束卷积
# ==========================================

def rgb_to_ycbcr(image):
    """
    将 RGB 图像转换为 YCbCr。
    输入: [B, 3, H, W] (RGB)
    输出: [B, 3, H, W] (YCbCr)
    """
    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]

    # 标准 JPEG 转换公式
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5

    return torch.stack([y, cb, cr], dim=1)

class BayarConv2d(nn.Module):
    """
    Bayar 约束卷积层 (修正版)。
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(BayarConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # === 修正点 1: 权重形状必须是 [out_channels, in_channels, k, k] ===
        self.kernel = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        
        # === 修正点 2: Mask 形状也要对应修改 ===
        self.mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        center = kernel_size // 2
        self.mask[:, :, center, center] = 0
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, x):
        # 1. 对权重应用掩码，将中心置零
        masked_kernel = self.kernel * self.mask
        
        # 2. 计算周围权重的和
        # sum over kernel dimensions (2, 3) -> 也就是 H 和 W 维度
        sum_weights = torch.sum(masked_kernel, dim=(2, 3), keepdim=True)
        
        # 3. 将中心位置设为 -sum，确保整体和为 0
        center = self.kernel_size // 2
        
        # 构造最终的卷积核
        final_kernel = masked_kernel.clone()
        
        # === 修正点 3: 赋值时的维度匹配 ===
        # sum_weights 的形状是 [out, in, 1, 1]，可以直接赋值给中心点
        final_kernel[:, :, center, center] = -sum_weights.squeeze(-1).squeeze(-1)
        
        # 执行卷积
        return F.conv2d(x, final_kernel, stride=self.stride, padding=self.padding)
    
# === 新增部分 1: 轻量级空间域特征提取器 ===
class SpatialExtractor(nn.Module):
    """
    组合拳提取器：YCbCr + BayarConv
    """
    def __init__(self, out_dim=128):
        super(SpatialExtractor, self).__init__()
        
        # 第一层：BayarConv (约束卷积)，捕捉微观残差
        # 输入 3 通道 (Y, Cb, Cr)，输出 32 通道噪声特征
        self.bayar_conv = BayarConv2d(3, 32, kernel_size=5, padding=2)
        
        # 后续层：普通 CNN 提取特征
        self.features = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 256 -> 128
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 128 -> 64
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 加入 Dropout 防止过拟合
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # 1. 颜色空间转换: RGB -> YCbCr
        # 这一步至关重要，分离亮度和色度伪影
        x = rgb_to_ycbcr(x)
        
        # 2. 约束卷积提取残差
        x = self.bayar_conv(x)
        
        # 3. 常规特征提取
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class AIDE_Model(nn.Module):

    def __init__(self, resnet_path, convnext_path):
        super(AIDE_Model, self).__init__()
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])
        # === 新增部分 2: 初始化空间分支 ===
        # 这里定义特征维度为 128，你可以根据显存情况调整（64或256）
        # 使用 128 维特征，配合 BayarConv
        self.spatial_branch = SpatialExtractor(out_dim=128)
        # ================================
       
        if resnet_path is not None:
            pretrained_dict = torch.load(resnet_path, map_location='cpu')
        
            model_min_dict = self.model_min.state_dict()
            model_max_dict = self.model_max.state_dict()
    
            for k in pretrained_dict.keys():
                if k in model_min_dict and pretrained_dict[k].size() == model_min_dict[k].size():
                    model_min_dict[k] = pretrained_dict[k]
                    model_max_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skipping layer {k} because of size mismatch")
        
        #输入维度2048对应 ResNet-50最后一层全局池化后的特征维度。256对应 ConvNeXt-XLarge最后一层全局池化后的特征维度。两组特征在通道维度上进行拼接（Concatenation）
        #1024 (隐藏层维度)：MLP 内部第一层会将特征投影到 1024 维的中间表示空间。
        #2 (输出维度)：代表分类的类别数。在图像取证或医学诊断中，这通常对应 “真/假” 或 “正常/病变” 的二分类任务。
        # self.fc = Mlp(2048 + 256 , 1024, 2)
        # === 修改部分 3: 调整 MLP 输入维度 ===
        # 2048 (SRM ResNet特征) + 256 (CLIP特征) + 128 (新增的Spatial特征)
        self.fc = Mlp(2048 + 256 + 128, 1024, 2)

        print("build model with convnext_xxl")
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained=convnext_path
        )

        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()

        self.openclip_convnext_xxl.eval()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #ConvNeXt-XXL 的输出维度高达 3072。这行代码定义了一个线性层，将 3072 维的高维特征压缩到 256 维。
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),

        )
        #requires_grad = False: 彻底关闭梯度计算。 这意味着冻结模型的所有参数，在训练过程中不会更新它们的权重。
        # 模型只学习 convnext_proj 里的线性映射，利用 XXL 模型强大的零样本（Zero-shot）特征提取能力来辅助分类。
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False

    def forward(self, x):

        b, t, c, h, w = x.shape

        x_minmin = x[:, 0] #[b, c, h, w]
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens = x[:, 4]

        # === 修改部分 4: 提取空间域特征 ===
        # 在 HPF 破坏颜色信息之前，将 RGB Patch 传入空间分支
        # 我们对4个 Patch 分别提取，然后取平均
        s_min = self.spatial_branch(x_minmin)
        s_max = self.spatial_branch(x_maxmax)
        s_min1 = self.spatial_branch(x_minmin1)
        s_max1 = self.spatial_branch(x_maxmax1)
        
        # 得到空间域特征向量 [B, 128]
        x_spatial = (s_min + s_max + s_min1 + s_max1) / 4
        # 建议加上 Normalize，防止某个分支特征数值过大主导梯度
        x_spatial = F.normalize(x_spatial, dim=1)
        # ================================

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        # 关键公式: 这里进行了一次在线重正化。因为 tokens 可能是按照 DINOv2 的标准归一化的，但 ConvNeXt-XXL 是基于 CLIP 标准训练的。这个公式将数据从 DINOv2 空间无缝转换到 CLIP 空间。
        # 特征提取: 提取出 (3072, 8, 8) 的高维空间特征。
        # 投影 (Projection): 经过全局平均池化（avgpool）和线性映射（convnext_proj），将 3072 维压缩为 256 维 的 x_0。
        with torch.no_grad():
            
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            ) #[b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)

        # 归一化 CLIP 特征
        x_0 = F.normalize(x_0, dim=1)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        # 特征平均: 将四个分支的输出进行平均，得到一个鲁棒性更强的噪声特征 x_1（维度通常为 2048）。
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4
        # 归一化 SRM 特征
        x_1 = F.normalize(x_1, dim=1)

        # 拼接: 将“大模型的通用视觉理解（x_0）”与“特定任务的噪声纹理（x_1）”强行结合。
        # 分类: 最终通过 self.fc 输出预测结果。
        # x = torch.cat([x_0, x_1], dim=1)

        # === 修改部分 5: 融合所有特征 ===
        # x_0: 语义特征 (256)
        # x_1: 噪声特征 (2048)
        # x_spatial: 空间特征 (128)
        x = torch.cat([x_0, x_1, x_spatial], dim=1)

        x = self.fc(x)

        return x

def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model

