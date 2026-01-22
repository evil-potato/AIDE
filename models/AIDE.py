import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.fft
import torchvision.transforms.functional as TF


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

# ==========================================
# Part 3: Z域分析组件 (针对 30 通道优化)
# ==========================================

class KernelEstimator(nn.Module):
    def __init__(self, kernel_size=5):
        super(KernelEstimator, self).__init__()
        self.kernel_size = kernel_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, kernel_size * kernel_size),
            nn.Tanh()
        )
    def forward(self, x):
        k = self.regressor(self.features(x))
        return k.view(x.size(0), 1, self.kernel_size, self.kernel_size)

class ResidualExtraction(nn.Module):
    def __init__(self, kernel_size):
        super(ResidualExtraction, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
    def forward(self, img, kernel):
        B, C, H, W = img.shape
        img_reshape = img.contiguous().view(1, B*C, H, W)
        kernel_repeat = kernel.repeat(1, C, 1, 1).view(B*C, 1, self.kernel_size, self.kernel_size)
        blurred = F.conv2d(img_reshape, kernel_repeat, padding=self.pad, groups=B*C)
        return img - blurred.view(B, C, H, W)

class DenseZScanLayer(nn.Module):
    def __init__(self, num_channels=30):
        super(DenseZScanLayer, self).__init__()
        self.num_channels = num_channels
        
        # === 核心修改: 生成 30 个 Z 域半径 ===
        # 范围覆盖 0.85 (强衰减) 到 1.15 (强发散)
        # 这相当于对 Z 平面进行密集的环形扫描
        self.radii = np.linspace(0.85, 1.15, num_channels) 

    def compute_z_slice(self, img, r):
        # img: [B, 1, H, W] (灰度残差)
        B, _, H, W = img.shape
        device = img.device
        
        # 径向权重 W = r^(-dist_from_center)
        y = torch.arange(H, device=device).view(1, 1, H, 1) - H//2
        x = torch.arange(W, device=device).view(1, 1, 1, W) - W//2
        dist = torch.sqrt(x**2 + y**2) / (H//2) # 归一化距离
        
        # 权重计算 (r > 1 放大边缘，r < 1 放大中心)
        # 使用 pow(r, -n) 形式
        weight = torch.pow(r, -dist * 10) 
        
        # 加权 + FFT
        fft = torch.fft.fft2(img * weight)
        fft_shift = torch.fft.fftshift(fft)
        spec = torch.log(torch.abs(fft_shift) + 1e-6)
        
        # 抑制中心直流分量 (DC Suppression)
        cy, cx = H // 2, W // 2
        spec[:, :, cy-1:cy+2, cx-1:cx+2] = 0

        return spec

    def forward(self, x):
        # x: Residual [B, 3, H, W] -> 转灰度 -> [B, 1, H, W]
        x_gray = torch.mean(x, dim=1, keepdim=True)
        
        specs = []
        for r in self.radii:
            s = self.compute_z_slice(x_gray, r)
            specs.append(s)
            
        # 拼接成 [B, 30, H, W] 以匹配 ResNet 输入
        out = torch.cat(specs, dim=1)
        
        # 归一化
        mean = out.mean(dim=(2, 3), keepdim=True)
        std = out.std(dim=(2, 3), keepdim=True)
        return (out - mean) / (std + 1e-5)

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
        # 1. 前端处理
        self.kernel_net = KernelEstimator(kernel_size=7)
        self.res_layer = ResidualExtraction(kernel_size=7)
        
        # 2. Z域扫描 (输出 30 通道)
        self.z_layer = DenseZScanLayer(num_channels=30)
        
        # 3. 你的 ResNet (作为 Backbone)
        # 使用 ResNet50 的配置 (Bottleneck, [3, 4, 6, 3])
        # 输入通道已在 CustomResNet 内部设为 30
        # 适配层 (30 -> 3): 让 Z 域特征能利用 ResNet 预训练权重
        self.adapter = nn.Conv2d(30, 3, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')

        # 使用标准 torchvision ResNet50
        self.backbone = models.resnet50(pretrained=False)

        # 加载 ResNet 权重
        if resnet_path:
            print(f"Loading ResNet weights from {resnet_path}")
            try:
                state_dict = torch.load(resnet_path, map_location='cpu')
                # 过滤不匹配的层 (如 fc)
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                self.backbone.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Error loading ResNet weights: {e}")
        else:
            print("Warning: resnet_path is None. Using random init.")

        # 移除 FC 层，只取特征
        self.backbone.fc = nn.Identity()

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

        #输入维度2048对应 ResNet-50最后一层全局池化后的特征维度。256对应 ConvNeXt-XLarge最后一层全局池化后的特征维度。两组特征在通道维度上进行拼接（Concatenation）
        #1024 (隐藏层维度)：MLP 内部第一层会将特征投影到 1024 维的中间表示空间。
        #2 (输出维度)：代表分类的类别数。在图像取证或医学诊断中，这通常对应 “真/假” 或 “正常/病变” 的二分类任务。
        # self.fc = Mlp(2048 + 256 , 1024, 2)

        # === [修改] FC输入维度 ===
        self.fc = Mlp(2048 + 256, 1024, 2)

    

    def forward(self, x):

        b, t, c, h, w = x.shape

        tokens = x[:, 4]

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


        # A. 估计核
        k = self.kernel_net(x[:, 4])
        
        # B. 提取残差
        res = self.res_layer(x[:, 4], k)
        
        # C. 30层 Z 域扫描 [B, 30, H, W]
        z_stack = self.z_layer(res)
        z_adapted = self.adapter(z_stack)# 4. 适配 (30->3通道)

        # D. ResNet 特征提取 [B, 2048]
        features = self.backbone(z_adapted)

        # 拼接: 将“大模型的通用视觉理解（x_0）”与“特定任务的噪声纹理（x_1）”强行结合。
        # 分类: 最终通过 self.fc 输出预测结果。
        x = torch.cat([x_0, features], dim=1)

        x = self.fc(x)

        # [关键修改] 训练模式下返回辅助信息
        if self.training:
            return x, k, res
        else:
            return x

def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model

