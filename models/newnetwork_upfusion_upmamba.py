import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import convnext
import convnext_isotropic as convnext_i  # 去掉三个下采样层的convnext
from mamba_ssm import Mamba
from SSM_my.mamba_my_simlpe import Mamba as BiConv_mamba
class ConVneXtAutoEncoder(nn.Module):
    def __init__(self, model):
        super(ConVneXtAutoEncoder, self).__init__()
        self.encoder = model
        self.flatten=nn.Flatten()
        self.fc = None
        # self.act = nn.ReLU()

    def forward(self, x):
        features = self.encoder(x)
        features=self.flatten(features)
        return self.fc(features)
        # return self.act(features)

    def pre_train(self,x,dim_high):
        features = self.encoder(x)
        _, C, H, W = features.shape
        if self.fc is None:
            self.fc = nn.Linear(C * H * W, dim_high).to(x.device)
        features = self.flatten(features)
        return self.fc(features), C, H, W


class ConVneXtAutoDecoder(nn.Module):
    def __init__(self, model):
        super(ConVneXtAutoDecoder, self).__init__()
        self.decoder = model
        self.unflatten=None
        self.fc=None
        # self.act = nn.ReLU()

    def forward(self, x,C,H,W):
        if  self.fc is None:
            self.fc=nn.Linear(x.shape[1],C*H*W).to(x.device)
            self.unflatten=nn.Unflatten(1,(C,H,W))
        feature=self.fc(x)
        feature=self.unflatten(feature)
        return self.decoder(feature)
        # return self.act(x)

class SpaMamba(nn.Module):
    def __init__(self, channels, num_state,use_residual=True, group_num=4, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = BiConv_mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=channels,  # Model dimension d_model
            d_state=num_state,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self, x):
        B, C,H,W = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_flat.view(B, H*W, C)
        x_flat = self.mamba(x_flat)
        x_flat = x_flat.view(B,H, W, C)
        x_flat = x_flat.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_flat = self.proj(x_flat)
        if self.use_residual:
            return x_flat + x
        else:
            return x_flat
class SpeMamba(nn.Module):
    def __init__(self, channels,num_state=16, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba =BiConv_mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.group_channel_num,  # Model dimension d_model
            d_state=num_state,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        x = self.padding_feature(x)
        x_flat = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_flat.shape
        x_flat = x_flat.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_flat = x_flat.view(B, H, W, C)
        x_flat = x_flat.permute(0, 3, 1, 2).contiguous()
        x_flat = self.proj(x_flat)
        if self.use_residual:
            return x + x_flat
        else:
            return x_flat
class Enhance(nn.Module):
    def __init__(self, channels,num_state=16, use_residual=True, group_num=4, use_proj=True):
        super(Enhance, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba( d_model=channels, 
            d_state=num_state, 
            d_conv=2,  
            expand=2,  
        )
        if self.use_proj:
            self.proj_hsi = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )
            # self.proj_lidar = nn.Sequential(
            #     nn.GroupNorm(group_num, channels),
            #     nn.SiLU()
            # )
    def forward(self, x_hsi):
        x_flat_hsi = x_hsi.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_flat_hsi.shape
        x_flat_hsi = x_flat_hsi.view(B, H*W, C)
        x_flat_hsi = self.mamba(x_flat_hsi)
        x_flat_hsi = x_flat_hsi.view(B, H, W, C)
        x_flat_hsi = x_flat_hsi.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_flat_hsi = self.proj_hsi(x_flat_hsi)
        if self.use_residual:
            return x_flat_hsi + x_hsi
        else:
            return x_flat_hsi
class cross_fuison(nn.Module):
    def __init__(self,input_size,output_size) :
        super(cross_fuison,self).__init__()
        self.W_Q_hsi = nn.Linear(input_size,output_size)
        self.W_K_hsi = nn.Linear(input_size,output_size)
        self.W_V = nn.Linear(input_size,output_size)
        self.W_Q_lidar = nn.Linear(input_size,output_size)
        self.W_K_lidar = nn.Linear(input_size,output_size)
        self.epsilon=1
        self.max_value=(output_size*4)**0.5
        self.scale_factor = torch.nn.Parameter(torch.tensor(output_size**0.5)) # 缩放因子，可学习的参数，初始化为输出维度的根号
        self.extand=nn.Linear(output_size,input_size)

    def forward(self, x1,x2):
        hsi_Q=self.W_Q_hsi(x1)
        hsi_K=self.W_K_hsi(x1)
        hsi_V=self.W_V(x1)
        lidar_Q=self.W_Q_lidar(x2)
        lidar_K=self.W_K_lidar(x2)
        lidar_V=self.W_V(x2)
        scale_factor=torch.clamp(self.scale_factor, min=self.epsilon, max=self.max_value)

        f_fusion1=F.softmax(torch.mm(hsi_Q, lidar_K.T)/(scale_factor ** 0.5),dim=-1)@hsi_V
        f_fusion2=F.softmax(torch.mm(lidar_Q, hsi_K.T) / (scale_factor ** 0.5),dim=-1) @lidar_V
        return self.extand(f_fusion1),self.extand(f_fusion2)


class BothMamba(nn.Module):
    def __init__(self, channels,output_size ,token_num, num_state,use_residual, group_num=4, use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        self.spa_mamba_hsi = SpaMamba(channels, num_state,use_residual=use_residual, group_num=group_num)
        self.spa_mamba_lidar = SpaMamba(channels, num_state,use_residual=use_residual, group_num=group_num)
        self.spe_mamba = SpeMamba(channels, num_state,token_num=token_num, use_residual=use_residual, group_num=group_num)
        self.enhance_hsi= Enhance(output_size, num_state, use_residual=use_residual, group_num=group_num)
        self.enhance_lidar = Enhance(output_size, num_state, use_residual=use_residual, group_num=group_num)
        self.cross_f=cross_fuison(output_size,int(output_size/2))

    def forward(self, x):
        B,C=x[0].shape
        x=list(x)
        x[0] = x[0].view(B, C, 1, 1)
        x[1] = x[1].view(B, C, 1, 1)
        spa_x_hsi = self.spa_mamba_hsi(x[0])
        spe_x = self.spe_mamba(spa_x_hsi)
        spa_x_lidar = self.spa_mamba_lidar(x[1])
        enhance_hsi=self.enhance_hsi(spe_x)
        enhance_lidar=self.enhance_lidar(spa_x_lidar)
        f_fusion1,f_fusion2=self.cross_f(spe_x.view(B, C),spa_x_lidar.view(B, C))   
        hsi_feature=(enhance_hsi.view(B,C))+f_fusion2
        lidar_feature=(enhance_lidar.view(B,C))+f_fusion1
        return [hsi_feature, lidar_feature]



class Net(nn.Module):
    def __init__(self, num_views, input_sizes, dim_high_feature, num_clusters,num_state=16,
                 token_num=4, group_num=4,L=1 ,use_att=True):
        super(Net, self).__init__()
        self.encouders = nn.ModuleList()
        self.decouders = nn.ModuleList()

        for idx in range(num_views):
            input_size = input_sizes[idx]
            convnext_model = convnext_i.ConvNeXtIsotropic(in_chans=input_size, dim=dim_high_feature)
            re_convnext_model = convnext_i.Re_ConvNeXt(in_chans=input_size, dim=dim_high_feature)
            self.encouders.append(ConVneXtAutoEncoder(convnext_model))
            self.decouders.append(ConVneXtAutoDecoder(re_convnext_model))
        #一个共享参数的MLP层将两个模态的特征映射到同一低维空间
        self.layers=nn.ModuleList()
        for i in range(L):
            self.layers.append(BothMamba(dim_high_feature,dim_high_feature,token_num,
                                         num_state,use_residual=True,group_num=group_num,use_att=use_att))
        # 使用一个共享参数的MLP去学习两个模态的标签分配，通过softmax函数得到聚类概率矩阵H
        # self.act=nn.SiLU()
        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, num_clusters),
            nn.Softmax(dim=1)
        )
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x= [self.encouders[0](x[0]), self.encouders[1](x[1])]#编码器映射后的特征
        for i,layer in enumerate(self.layers):
            x = layer(x)
        labels_pro = [self.label_learning_module(x[0]), self.label_learning_module(x[1])]
        labels=(labels_pro[0]+labels_pro[1])/2

        labels=torch.argmax(labels,dim=1)
        return labels_pro, x,labels


    def pre_train(self,x, dim_high):
        re_features = list()
        for idx in range(len(x)):
            data_view = x[idx]
            feature, C, H, W = self.encouders[idx].pre_train(data_view,dim_high)
            re_feature = self.decouders[idx](feature, C, H, W)
            re_features.append(re_feature)
        return re_features
