import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import torch.nn.functional as F
import numpy as np

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

# 增加交叉注意力融合模块
class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim=256, key_dim=2048, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 将不同维度的特征投影到相同的嵌入空间
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        
        self.att = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # 最后的输出投影，用于保持特征多样性
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query_feat, key_feat):
        # query_feat: [b, 256], key_feat: [b, 2048]
        # 增加序列维度以符合 MultiheadAttention 要求: [b, 1, dim]
        q = self.q_proj(query_feat).unsqueeze(1) 
        k = self.k_proj(key_feat).unsqueeze(1)
        v = self.v_proj(key_feat).unsqueeze(1)
        
        # 交叉注意力计算
        attn_out, _ = self.att(q, k, v)
        
        # 还原形状并归一化
        out = self.norm(self.out_proj(attn_out.squeeze(1)))
        return out

# 增加门控融合模块
class GatedFusion(nn.Module):
    def __init__(self, dim_b=256, dim_a=2048, dim_c=2048):
        super().__init__()
        # 总维度：256 + 2048 + 2048 = 4352
        total_dim = dim_b + dim_a + dim_c
        
        # 门控网络：计算三路特征的相对重要性
        self.gate = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Linear(512, 3), # 为三个分支分别输出一个权重
            nn.Softmax(dim=1)
        )
        
    def forward(self, x_b, x_a, x_c):
        # x_b: ConvNeXt [b, 256]
        # x_a: HPF-ResNet [b, 2048]
        # x_c: Raw-ResNet [b, 2048]
        
        combined = torch.cat([x_b, x_a, x_c], dim=1) # [b, 4352]
        weights = self.gate(combined) # [b, 3]
        
        # 应用权重：每个分支乘以其对应的门控权值
        out_b = x_b * weights[:, 0:1]
        out_a = x_a * weights[:, 1:2]
        out_c = x_c * weights[:, 2:3]
        
        # 再次拼接作为最终融合特征
        return torch.cat([out_b, out_a, out_c], dim=1)

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

    # def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True, in_channels=30):
        super(ResNet, self).__init__()

        self.inplanes = 64
        # self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3,
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
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
        self.model_space = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=3)

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
        
        # 交叉注意力模块
        # 让 x0 分别与 x1 和 x2 进行交叉
        # self.cross_attn_1 = CrossAttentionFusion(query_dim=256, key_dim=2048, embed_dim=512)
        # self.cross_attn_2 = CrossAttentionFusion(query_dim=256, key_dim=2048, embed_dim=512)
        # 融合后的维度：
        # x0 (256) + 交叉特征1 (512) + 交叉特征2 (512)
        # total_fusion_dim = 256 + 512 + 512
        
        #门控融合模块
        # self.fusion_layer = GatedFusion(dim_b=256, dim_a=2048, dim_c=2048)
        total_fusion_dim = 256 + 2048 + 2048

        # 定义辅助分类器 (Auxiliary Classifiers)
        # x_1 的维度是 2048, x_2 的维度是 2048
        self.aux_head_noise = Mlp(2048, 512, 2)
        self.aux_head_space = Mlp(2048, 512, 2)

        self.fc = Mlp(total_fusion_dim, 1024, 2)

        #输入维度2048对应 ResNet-50最后一层全局池化后的特征维度。256对应 ConvNeXt-XLarge最后一层全局池化后的特征维度。两组特征在通道维度上进行拼接（Concatenation）
        #1024 (隐藏层维度)：MLP 内部第一层会将特征投影到 1024 维的中间表示空间。
        #2 (输出维度)：代表分类的类别数。在图像取证或医学诊断中，这通常对应 “真/假” 或 “正常/病变” 的二分类任务。
        # self.fc = Mlp(2048 + 256 + 2048, 1024, 2)

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

        # 先冻结频域模型和辅助分类头，突击训练空间域模型
        for param in self.model_min.parameters():
            param.requires_grad = True
        for param in self.model_max.parameters():
            param.requires_grad = True
        for param in self.aux_head_noise.parameters():
            param.requires_grad = True
        for param in self.model_space.parameters():
            param.requires_grad = False
        for param in self.aux_head_space.parameters():
            param.requires_grad = False

    

    def forward(self, x):

        b, t, c, h, w = x.shape

        x_minmin = x[:, 0] #[b, c, h, w]
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens = x[:, 4]

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

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        # 特征平均: 将四个分支的输出进行平均，得到一个鲁棒性更强的噪声特征 x_1（维度通常为 2048）。
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4

        #增加空间域特征
        x_2 = self.model_space(x[:, 4])

        # 核心：在训练模式下返回辅助预测值
        if self.training:
            aux_noise_out = self.aux_head_noise(x_1)
            aux_space_out = self.aux_head_space(x_2)

        # 执行交叉注意力融合
        # 使用大模型语义特征 x_0 去“筛选”残差特征 x_1 和 原始特征 x_2
        # 得到两个融合后的特征 feat_1 和 feat_2（维度均为 512）。
        # feat_1 = self.cross_attn_1(x_0, x_1)  # [b, 512]
        # feat_2 = self.cross_attn_2(x_0, x_2)  # [b, 512]

        # 拼接: 将“大模型的通用视觉理解（x_0）”与“（feat_1）”和“（feat_2）”在通道维度上进行拼接，形成一个综合特征向量。
        # 分类: 最终通过 self.fc 输出预测结果。
        # x = torch.cat([x_0, feat_1, feat_2], dim=1)

        #执行门控融合
        # x= self.fusion_layer(x_0, x_1, x_2) # [b, 4352]

        x = torch.cat([x_0, x_1, x_2], dim=1)
        x = self.fc(x)

        if self.training:
            # --- 正交损失 (Orthogonality Loss) ---
            # 1. 对特征进行 L2 归一化
            proj_1 = F.normalize(x_1, p=2, dim=1) # [b, 2048]
            proj_2 = F.normalize(x_2, p=2, dim=1) # [b, 2048]
            
            # 2. 计算余弦相似度的绝对值
            # torch.sum(proj_1 * proj_2, dim=1) 得到每个样本的相似度
            # 绝对值越接近 0，表示越正交
            ortho_loss = torch.mean(torch.abs(torch.sum(proj_1 * proj_2, dim=1)))
            
            # 返回主输出和两个辅助输出
            return x, aux_noise_out, aux_space_out, ortho_loss
        else:
            return x

        # return x

def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model

