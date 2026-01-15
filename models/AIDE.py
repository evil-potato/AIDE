import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np
import torch.nn.functional as F
import torch.fft
import torchvision.transforms.functional as TF


# ==========================================
# [新增] 频谱特征提取器 (Spectral Extractor)
# ==========================================
class SpectralExtractor(nn.Module):
    """
    输入: RGB Patch [B, 3, H, W]
    输出: 频域特征向量 [B, out_dim]
    作用: 提取 Patch 的频域指纹，用于后续的一致性校验。
    """
    def __init__(self, out_dim=128):
        super(SpectralExtractor, self).__init__()
        
        # 定义处理频谱图的浅层 CNN
        # 频谱图具有中心对称性且能量集中在低频，使用大核卷积捕捉全局分布
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Global Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(p=0.5) # 强力 Dropout 防止过拟合
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # 1. 快速傅里叶变换 (RFFT)
        # 输入是实数，使用 rfft2 计算只返回非冗余的一半频谱，效率更高
        # 输出形状 [B, 3, H, W/2 + 1] (复数)
        fft = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 取幅值 (Modulus)
        fft_abs = torch.abs(fft)
        
        # 3. 对数变换 (Log Transform)
        # 关键步骤：压缩动态范围，让模型能看清高频的微弱信号
        # 加 1e-8 防止 log(0)
        fft_log = torch.log(fft_abs + 1e-8)
        
        # 4. CNN 特征提取
        x = self.conv(fft_log)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==========================================
# [新增] 语义引导门控 (Semantic Gating)
# ==========================================
class SemanticGate(nn.Module):
    """
    输入: CLIP 语义特征 [B, sem_dim]
    输出: 频域注意力权重 [B, freq_dim]
    作用: 根据内容（如"人脸"或"风景"）动态调整关注的频段
    """
    def __init__(self, sem_dim=256, freq_dim=128):
        super(SemanticGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sem_dim, sem_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(sem_dim // 2, freq_dim),
            nn.Sigmoid() # 输出 0~1 之间的权重
        )

    def forward(self, semantic_feat):
        # 生成门控权重
        # semantic_feat: [B, 256] -> gate: [B, 128]
        gate = self.mlp(semantic_feat)
        return gate
    

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
       
        # === [新增] 初始化新模块 ===
        self.spectral_branch = SpectralExtractor(out_dim=128)
        self.sem_gate = SemanticGate(sem_dim=256, freq_dim=128)
        # =========================

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

        # === [修改] FC输入维度 ===
        # 2048(SRM) + 256(CLIP) + 128(Freq Mean) + 128(Freq Std)
        self.fc = Mlp(2048 + 256 + 128 + 128, 1024, 2)

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

        # =========================================================
        # [Step 1] CLIP 语义提取 (提前执行，为 Gate 做准备)
        # =========================================================
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

        # 归一化语义特征，供后续融合和Gate使用
        x_0_norm = F.normalize(x_0, dim=1)

        # =========================================================
        # [Step 2] 频谱一致性分支 (使用 RGB Patch)
        # =========================================================
        # 定义增强函数 (仅针对 FFT 分支)
        def fft_aug(img):
            if self.training: 
                if torch.rand(1) < 0.5: # 随机模糊
                    sigma = float(torch.rand(1) * 1.9 + 0.1)
                    img = TF.gaussian_blur(img, kernel_size=5, sigma=[sigma, sigma])
                if torch.rand(1) < 0.5: # 随机噪声
                    img = img + torch.randn_like(img) * 0.02
            return img

        # 提取频谱特征 (使用增强后的 patch)
        f_min = self.spectral_branch(fft_aug(x_minmin))
        f_max = self.spectral_branch(fft_aug(x_maxmax))
        f_min1 = self.spectral_branch(fft_aug(x_minmin1))
        f_max1 = self.spectral_branch(fft_aug(x_maxmax1))

        # 堆叠 [B, 4, 128]
        freq_stack = torch.stack([f_min, f_max, f_min1, f_max1], dim=1)

        # 语义门控: 生成权重并加权
        gate_weights = self.sem_gate(x_0_norm) # [B, 128]
        gate_weights = gate_weights.unsqueeze(1) # [B, 1, 128]
        freq_refined = freq_stack * gate_weights

        # 计算一致性
        freq_mean = torch.mean(freq_refined, dim=1) # 信号特征
        freq_std = torch.std(freq_refined, dim=1)   # 不一致性特征 (Sanity Check)

        freq_mean = F.normalize(freq_mean, dim=1)
        freq_std = F.normalize(freq_std, dim=1)

        # =========================================================
        # [Step 3] SRM 噪声分支 (原逻辑，使用 HPF)
        # =========================================================
        # 注意: 这里的 x_minmin 变量会被覆盖为噪声图，所以必须在 Step 2 之后执行
        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        # 特征平均: 将四个分支的输出进行平均，得到一个鲁棒性更强的噪声特征 x_1（维度通常为 2048）。
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4
        x_1 = F.normalize(x_1, dim=1) # 加上归一化更稳健

        # 拼接: 将“大模型的通用视觉理解（x_0）”与“特定任务的噪声纹理（x_1）”强行结合。
        # 分类: 最终通过 self.fc 输出预测结果。
        # x = torch.cat([x_0, x_1], dim=1)
        # 拼接: 语义(256) + 噪声(2048) + 频谱均值(128) + 频谱不一致性(128)
        x = torch.cat([x_0, x_1, freq_mean, freq_std], dim=1)

        x = self.fc(x)

        return x

def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model

