import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score, mean_squared_error

# 运行改进版的HindmarshRose, CoupledRossler  jia iF

class GraphSAGE(nn.Module):
    def __init__(self, feature_dim, ode_hid_dim):
        super(GraphSAGE, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(feature_dim, ode_hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ode_hid_dim, ode_hid_dim, bias=True),
        )
        self.f2 = nn.Sequential(
            nn.Linear(ode_hid_dim, ode_hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ode_hid_dim, feature_dim, bias=True),
        )

        self.adj = None  # Adjacency matrix for neighborhood aggregation

    def forward(self, x):
        # First feature transformation
        x = self.f1(x)

        # GraphSAGE neighborhood aggregation (mean aggregator)
        if self.adj is not None:
            neighbor_feats = torch.matmul(self.adj, x)  # Aggregate neighbor features
            x = torch.cat([x, neighbor_feats], dim=-1)  # Concatenate self and neighbor features

        # Apply second feature transformation
        x = self.f2(x)
        return x

class FullAttention(nn.Module):
    """
    FullAttention implements a scaled dot-product attention with optional sparsity factors
    (for example, to approximate attention in long sequences).
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor  # Sparsity factor for approximating attention
        self.scale = scale  # Scaling factor for dot product
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries: Tensor of shape (batch_size, num_heads, query_len, d_k)
        keys: Tensor of shape (batch_size, num_heads, key_len, d_k)
        values: Tensor of shape (batch_size, num_heads, key_len, d_v)
        attn_mask: Optional mask to exclude certain positions from the attention
        """
        B, H, L_Q, D = queries.size()
        _, _, L_K, _ = keys.size()

        # Scaled dot-product attention
        scale = self.scale or 1.0 / (D ** 0.5)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Output
        out = torch.matmul(attention, values)

        if self.output_attention:
            return out, attention
        else:
            return out, None




import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import mercator
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import torchdiffeq as ode

from utils import drawTraj, draw_embedding


def normalized_laplacian(A: torch.Tensor):
    """Symmetrically Normalized Laplacian: I - D^-1/2 * ( A ) * D^-1/2"""
    out_degree = torch.sum(A, dim=1)
    int_degree = torch.sum(A, dim=0)

    out_degree_sqrt_inv = torch.pow(out_degree, -0.5)
    int_degree_sqrt_inv = torch.pow(int_degree, -0.5)
    mx_operator = torch.eye(A.shape[0], device=A.device) - torch.diag(out_degree_sqrt_inv) @ A @ torch.diag(
        int_degree_sqrt_inv)

    return mx_operator


class HyperbolicEmbedding:
    def __init__(self, args: dict):
        self.args = args

    def fit_transform(self):
        if not os.path.exists(f'{self.args.log_dir}/HE/he.inf_coord'):
            os.makedirs(f'{self.args.log_dir}/HE', exist_ok=True)
            mercator.embed(
                edgelist_filename=f'{self.args.data_dir}/graph.txt',
                quiet_mode=self.args.quiet_mode,
                fast_mode=self.args.fast_mode,
                output_name=f'{self.args.log_dir}/HE/he',
                validation_mode=self.args.validation_mode,
                post_kappa=self.args.post_kappa
            )

        if self.args.refine:
            mercator.embed(
                edgelist_filename=f'{self.args.data_dir}/graph.txt',
                quiet_mode=self.args.quiet_mode,
                fast_mode=self.args.fast_mode,
                output_name=f'{self.args.log_dir}/HE/he',
                validation_mode=self.args.validation_mode,
                post_kappa=self.args.post_kappa,
                inf_coord=f'{self.args.log_dir}/HE/he.inf_coord'
            )

        return self._parse_mercator_output()

    def _parse_mercator_output(self):
        with open(f'{self.args.log_dir}/HE/he.inf_coord', 'r') as f:
            lines = f.readlines()

        # parse node_num, dim, coords
        node_num = int(lines[7].split()[-1])
        beta = float(lines[8].split()[-1])
        mu = float(lines[9].split()[-1])

        kappa = np.zeros(node_num)
        angular = np.zeros(node_num)
        radius = np.zeros(node_num)
        for i in range(15, 15 + node_num):
            kappa[i - 15] = float(lines[i].split()[1])
            angular[i - 15] = float(lines[i].split()[2])
            radius[i - 15] = float(lines[i].split()[3])

        return kappa, angular, radius


def atanh(x, eps=1e-5):
    x = torch.clamp(x, max=1. - eps)
    return .5 * (torch.log(1 + x) - torch.log(1 - x))


class PoincareManifold:

    @staticmethod
    def poincare_grad(euclidean_grad, x, c=-1, eps=1e-5):
        """
        Compute the gradient of the Poincare distance with respect to x.
        """
        sqnormx = torch.sum(x * x, dim=-1, keepdim=True)
        result = ((1 + c * sqnormx) / 2) ** 2 * euclidean_grad
        return result

    @staticmethod
    def log_map_zero(x, c=-1, eps=1e-5):
        """
        Log map from Poincare ball space to tangent space of zero.
        Ref:
        1. https://github.com/cll27/pvae/tree/7abbb4604a1acec2332b1b4dfe21267834b505cc
        2. https://github.com/facebookresearch/hgnn/blob/master/manifold/PoincareManifold.py
        """
        norm_diff = torch.norm(x, 2, dim=1, keepdim=True)
        atanh_x = atanh(np.sqrt(np.abs(c)) * norm_diff)
        lam_zero = 2.  # lambda = 2 / (1 + ||zero||) = 2
        return 2. / (np.sqrt(np.abs(c)) * lam_zero) * atanh_x * (x + eps) / norm_diff


class GNN(nn.Module):
    def __init__(self, feature_dim, ode_hid_dim):
        super(GNN, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(feature_dim, ode_hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ode_hid_dim, ode_hid_dim, bias=True),
        )
        self.f2 = nn.Sequential(
            nn.Linear(ode_hid_dim, ode_hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ode_hid_dim, feature_dim, bias=True),
        )

        self.adj = None

    def forward(self, x):
        x = self.f1(x)
        x = self.adj @ x
        x = self.f2(x)
        return x



# # 改
class InvertedAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(InvertedAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Input: B x D x L (batch, variable, time)
        queries = self.query_projection(x)  # B x D x L
        keys = self.key_projection(x)       # B x D x L
        values = self.value_projection(x)   # B x D x L

        # Attention mechanism
        out, _ = self.inner_attention(queries, keys, values)

        # Output projection
        return self.out_projection(out)  # B x D x L

class FFCMAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, ffcm_dim):
        super(FFCMAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

        # Add FFCM for frequency processing
        self.ffcm = FFCM(dim=ffcm_dim, token_mixer_for_gloal=Freq_Fusion)

    def forward(self, x):
        # Input: B x D x L (batch, variable, time)
        x_ffcm = self.ffcm(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        queries = self.query_projection(x_ffcm)
        keys = self.key_projection(x_ffcm)
        values = self.value_projection(x_ffcm)

        # Attention mechanism
        out, _ = self.inner_attention(queries, keys, values)

        # Output projection
        return self.out_projection(out)  # B x D x L

class WindowedAttention(nn.Module):
    def __init__(self, attention_layer, window_size):
        super(WindowedAttention, self).__init__()
        self.attention_layer = attention_layer
        self.window_size = window_size

    def forward(self, x):
        # Split input into windows along the time dimension
        B, D, L = x.size()
        num_windows = L // self.window_size
        x = x.unfold(dimension=-1, size=self.window_size, step=self.window_size)  # B x D x num_windows x window_size

        # Apply attention to each window
        x = x.permute(0, 2, 1, 3).contiguous()  # B x num_windows x D x window_size
        x = self.attention_layer(x)

        # Merge windows back
        x = x.view(B, D, -1)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        # Perform 2D FFT
        ffted = torch.fft.rfft2(x, norm="ortho")
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # Reshape for convolution
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous().view(batch, -1, h, w // 2 + 1)
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        # Reshape back to complex and inverse FFT
        ffted = ffted.view(batch, -1, 2, h, w // 2 + 1).permute(0, 1, 3, 4, 2)



        # print(f"ffted.shape: {ffted.shape}")
        # print(f"ffted.stride(): {ffted.stride()}")
        # 调试信息：检查 ffted 的形状和 stride
        # print(f"Before view_as_complex: ffted.shape = {ffted.shape}, ffted.stride = {ffted.stride()}")

        # 确保最后一维大小为 2
        if ffted.shape[-1] != 2:
            raise ValueError("Input tensor for `view_as_complex` must have last dimension of size 2.")

        # 确保内存布局连续
        ffted = ffted.contiguous()

        # 转换为复数
        ffted = torch.view_as_complex(ffted)

        # 调试信息：查看转换后的形状
        # print(f"After view_as_complex: ffted.shape = {ffted.shape}")



        # ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm="ortho")
        return output


class Freq_Fusion(nn.Module):
    def __init__(self, dim, kernel_size=[1, 3, 5, 7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim * scale_ratio // spilt_num
        self.conv_init_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU()
        )
        self.FFC = FourierUnit(in_channels=dim * 2, out_channels=dim * 2)
        self.bn = nn.BatchNorm2d(dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class FFCM(nn.Module):
    def __init__(self, dim, token_mixer_for_gloal=Freq_Fusion, mixer_kernel_size=[1, 3, 5, 7], local_size=8):
        super(FFCM, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size, se_ratio=8, local_size=local_size)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode="reflect"),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
         nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode="reflect"),
            # nn.Conv2d(dim, dim, kernel_size=1, padding=1 // 2, groups=dim, padding_mode="reflect"),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=2, padding=2 // 2, groups=dim, padding_mode="reflect"),
             nn.Conv2d(dim, dim, kernel_size=5, padding=5 // 2, groups=dim, padding_mode="reflect"),
            nn.GELU()
        )
        self.FFC = FourierUnit(dim * 2, dim * 2)  # 假设FourierUnit类已正确定义且在当前作用域可访问，这里添加实例化
        self.bn = nn.BatchNorm2d(dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(f"输入x的原始形状: {x.shape}")
        batch_size, seq_len, feature_dim = x.shape[0], x.shape[1], x.shape[2]

        # 调整维度，确保只在通道维度进行填充，使其变为符合conv2d要求的4D张量（批量情况）
        if feature_dim!= 64:
            # 创建一个新的张量，在通道维度进行填充
            new_x = torch.zeros((batch_size, 64, seq_len, 1), device=x.device)
            # 根据unsqueeze后的实际维度数量调整permute参数，确保维度顺序调整正确
            # x_expanded = x.unsqueeze(-1).unsqueeze(-1)
            # 确保形状匹配目标
            x_expanded = x.permute(0, 2, 1, 3, 4).contiguous()  # 直接调整维度顺序，保持 5 维

            # 变为6维，例如 [batch_size, seq_len, feature_dim, 1, 1]
            # print(f"x_expanded.shape: {x_expanded.shape}")
            # new_x[:, :feature_dim, :, :] = x_expanded.permute(0, 2, 1, 3, 4).contiguous()  # 先调整维度顺序使其和new_x的维度对应上，这里按照6维情况设置参数并确保内存连续

            new_x[:, :feature_dim, :, :] = x_expanded.squeeze(-1)
            x = new_x
            # print(f"填充后x的形状（使通道数达到64）: {x.shape}")
        else:
            x = x.unsqueeze(-1).unsqueeze(1)  # 调整为 [batch_size, feature_dim, seq_len, 1] 形式
            # print(f"调整视图后x的形状（通道数为64时）: {x.shape}")

        # 应用全局傅里叶单元
        # print(f"进入conv_init前x的形状: {x.shape}")
        x = self.conv_init(x)
        # print(f"conv_init后x的形状: {x.shape}")
        x = self.FFC(x) + x
        # print(f"FFC操作后x的形状: {x.shape}")
        x = self.relu(self.bn(x))
        # print(f"bn和relu操作后x的形状: {x.shape}")

        # 移除额外的维度并恢复原来的维度顺序
        x = x.squeeze(-1).squeeze(1).view(batch_size, seq_len, -1)  # [batch_size, seq_len, feature_dim]
        # print(f"最终输出x的形状: {x.shape}")

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

# FullAttention Implementation
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask=None):
        B, H, L_Q, D = queries.size()
        _, _, L_K, _ = keys.size()

        # Scaled Dot-Product Attention
        scale = self.scale or 1.0 / (D ** 0.5)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, values)
        if self.output_attention:
            return out, attention
        else:
            return out, None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()

        # self.d_model = d_model  # 这里存储d_model的值
        # self.feature_projection = nn.Linear(3, 64)  # 从特征维度3到64
        #
        # self.inner_attention = attention
        # self.n_heads = n_heads
        # self.head_dim = d_model // n_heads
        #
        # self.query_projection = nn.Linear(64, d_model)  # 将64维投影至d_model
        # self.key_projection = nn.Linear(64, d_model)
        # self.value_projection = nn.Linear(64, d_model)
        # self.out_projection = nn.Linear(d_model, d_model)

        self.d_model = d_model  # 这里存储d_model的值
        self.feature_projection = nn.Linear(3, 16)  # 从特征维度3到64
        self.inner_attention = attention
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_projection = nn.Linear(16, d_model)  # 将64维投影至d_model
        self.key_projection = nn.Linear(16, d_model)
        self.value_projection = nn.Linear(16, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        x = self.feature_projection(x)  # 将输入特征转换

        queries = self.query_projection(x).view(x.shape[0], -1, self.n_heads, self.head_dim)
        keys = self.key_projection(x).view(x.shape[0], -1, self.n_heads, self.head_dim)
        values = self.value_projection(x).view(x.shape[0], -1, self.n_heads, self.head_dim)

        out, _ = self.inner_attention(queries, keys, values, attn_mask)

        out = out.view(x.shape[0], -1, self.d_model)
        return self.out_projection(out)

class FFCMAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, ffcm_dim):
        super(FFCMAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

        # FFCM for Frequency Processing
        self.ffcm = FFCM(dim=ffcm_dim, token_mixer_for_gloal=Freq_Fusion)

    def forward(self, x):
        # Apply FFCM
        x_ffcm = self.ffcm(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        queries = self.query_projection(x_ffcm)
        keys = self.key_projection(x_ffcm)
        values = self.value_projection(x_ffcm)

        out, _ = self.inner_attention(queries, keys, values)
        return self.out_projection(out)


# BackboneODE with Integrated Attention and FFCM
class BackboneODE(nn.Module):
    def __init__(self, lookback, feature_dim, ode_hid_dim, method, ffcm_dim, d_model, n_heads, window_size, dropout):
        super(BackboneODE, self).__init__()
        self.method = method
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 添加线性变换层
        # self.project_to_64_self = nn.Linear(feature_dim, 64)  # 确保输出维度为64
        # self.project_to_64_attention = nn.Linear(64, 64)  # 同样是64维
        # self.project_to_64_ffcm = nn.Linear(128, 64)  # 从128映射到64
        self.project_to_64_self = nn.Linear(feature_dim, 16)  # 确保输出维度为64
        self.project_to_64_attention = nn.Linear(16, 16)  # 同样是64维
        self.project_to_64_ffcm = nn.Linear(128, 16)  # 从128映射到64


        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.project_to_GNN = nn.Linear(3, 64).to(self.device)
        # self.mlp_layer1 = nn.Linear(64, 64).to(self.device)
        # self.mlp_layer2 = nn.Linear(64, 64).to(self.device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_to_GNN = nn.Linear(3, 16).to(self.device)
        self.mlp_layer1 = nn.Linear(16, 16).to(self.device)
        self.mlp_layer2 = nn.Linear(16, 16).to(self.device)

        # Initial Encoder
        self.init_enc = nn.Sequential(
            nn.Linear(lookback, ode_hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ode_hid_dim, 1, bias=True)
        )

        # Dynamics
        self.f = nn.Sequential(
            nn.Linear(feature_dim, ode_hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ode_hid_dim, feature_dim, bias=True),
        )



        # Attention Layer
        # self.attention_layer = AttentionLayer(
        #     FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
        #     d_model=d_model,
        #     n_heads=n_heads
        # )

        self.attention_layer = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
            # d_model=64,  # 统一为64维
            d_model=16,  # 统一为64维
            n_heads=n_heads
        )
        # GNN layer
        self.g = GraphSAGE(feature_dim, ode_hid_dim)
        # FFCM
        self.ffcm = FFCM(dim=ffcm_dim, token_mixer_for_gloal=Freq_Fusion)



        # Placeholder for adjacency weights
        self.adj_w = None

        # self.project_back_to_3d = nn.Linear(256, 3)
        # self.project_back_to_3d = nn.Linear(64, 3)
        self.project_back_to_3d = nn.Linear(16, 3)
    def dxdt(self, t, x):
        """
        Calculate dx/dt based on the current state and adjacency weights.
        """
        if self.adj_w is None:
            raise ValueError("adj_w is not set. Call forward() with valid adj_w before using dxdt.")

        # 调整 x 的形状
        # batch_size, time_steps, nodes, features = x.shape  # 假设 x 的形状是 (8, 12, 3892, 3)
        # x = x.view(batch_size * time_steps, nodes, features)  # 调整为 (96, 3892, 3)

        x_self = self.f(x)
        x_self = self.project_to_64_self(x_self)  # 变换到256维度

        x_attention = self.attention_layer(x)
        x_attention =self.project_to_64_attention(x_attention)  # 已经是256维度，无需变换

        x_ffcm = self.ffcm(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        x_ffcm = self.project_to_64_ffcm(x_ffcm)  # 变换到256维度

        # print(f"x_self.shape: {x_self.shape}")
        # print(f"x_attention.shape: {x_attention.shape}")
        # print(f"x_ffcm.shape: {x_ffcm.shape}")
        # 邻居动态特性计算
        x_neigh = self.g(x).to(self.device)
        x_neigh = x_neigh.view(-1, x_neigh.shape[-1]).to(self.device)
        x_neigh = self.project_to_GNN(x_neigh).to(self.device)
        x_neigh = x_neigh.view(x.shape[0], x.shape[1], -1).to(self.device)

        dxdt = x_self + x_attention + x_ffcm + x_neigh
        # dxdt = x_self + x_attention
        # 将结果变换回原始维度3
        dxdt = self.project_back_to_3d(dxdt)


        # return dxdt
        if dxdt.shape != x.shape:
            raise ValueError(f"dxdt shape mismatch: expected {x.shape}, got {dxdt.shape}")
        return dxdt

    def forward(self, tspan, x, adj_w):
        """
        Forward pass of BackboneODE with time span and adjacency weights.
        """
        # Store adjacency weights for use in dxdt
        self.adj_w = adj_w

        # Encode Initial State
        x = x.permute(0, 2, 3, 1)
        x = self.init_enc(x).squeeze(-1)


        # Solve ODE
        out = ode.odeint(self.dxdt, x, tspan, method=self.method)
        out = out.permute(1, 0, 2, 3)

        # Clear adj_w after computation
        self.adj_w = None

        return out

import math
class Refiner(nn.Module):
    def __init__(self, lookback, horizon, feature_dim, hid_dim):
        super(Refiner, self).__init__()

        self.feature_dim = feature_dim
        self.mlp_X = nn.Sequential(
            nn.Linear(lookback * feature_dim, hid_dim),
            nn.Tanh(),
        )
        self.mlp_Y = nn.Sequential(
            nn.Linear(horizon * feature_dim, hid_dim),
            nn.Tanh(),
        )

        # KAN structure
        self.kan_layers = nn.ModuleList([
            KANLinear(hid_dim * 2, 64),  # hidden layer size
        ])
        self.final_layer = nn.Linear(64, horizon * feature_dim)

    def forward(self, X, Y):
        X = X.permute(0, 2, 1, 3)  # batch_size, node_num, lookback, feature_dim
        X = X.reshape(X.shape[0], X.shape[1], -1)  # batch_size, node_num, lookback*feature_dim
        Y = Y.permute(0, 2, 1, 3)  # batch_size, node_num, horizon, feature_dim
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1)  # batch_size, node_num, horizon*feature_dim

        X = self.mlp_X(X)
        Y = self.mlp_Y(Y)

        output = torch.cat([X, Y], dim=-1)

        # Pass through KAN layers
        for layer in self.kan_layers:
            output = layer(output)

        refined_Y = self.final_layer(output)  # Output layer to project back

        refined_Y = refined_Y.reshape(refined_Y.shape[0], refined_Y.shape[1], -1,
                                      self.feature_dim)  # batch_size, node_num, horizon, feature_dim
        refined_Y = refined_Y.permute(0, 2, 1, 3)  # batch_size, horizon, node_num, feature_dim
        return refined_Y


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5,
                 spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True,
                 base_activation=torch.nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 确定设备：使用可用的 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create grid for B-splines并将其移动到 GPU
        self.grid = self.create_grid(grid_size, grid_range, spline_order).to(self.device)

        # Parameters并将其移动到 GPU
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features).to(self.device))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order).to(self.device))
        self.spline_scaler = torch.nn.Parameter(
            torch.Tensor(out_features, in_features).to(self.device)) if enable_standalone_scale_spline else None

        # Activation and Scaling
        self.base_activation = base_activation()

        print("Parameters initialized on device:")
        print(f"base_weight device: {self.base_weight.device}")
        print(f"spline_weight device: {self.spline_weight.device}")
        if self.spline_scaler is not None:
            print(f"spline_scaler device: {self.spline_scaler.device}")

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.grid_eps = grid_eps

        self.reset_parameters()

    def create_grid(self, grid_size, grid_range, spline_order):
        h = (grid_range[1] - grid_range[0]) / grid_size
        # 如果没有 GPU，使用 CPU
        device = self.device
        return (torch.arange(-spline_order, grid_size + spline_order + 1, device=device) * h + grid_range[0]).expand(
            self.in_features, -1).contiguous()

    def reset_parameters(self):
        # Initialize parameters in GPU
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features, device=self.device) - 0.5) * self.scale_noise / self.grid_size
        self.spline_weight.data.copy_(
            self.scale_spline * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
        if self.spline_scaler is not None:
            torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        # print(f"x device: {x.device}, grid device: {grid.device}")  # Debugging
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = self.basis_update(x, grid, bases, k)
        return bases.contiguous()

    def basis_update(self, x, grid, bases, k):
        return (
                (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] +
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-(k)]) * bases[:, :, 1:]
        )

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        # 确保 A 和 B 在相同的设备上
        solution = torch.linalg.lstsq(A.to(self.device), B.to(self.device)).solution  # Least squares solution
        return solution.permute(2, 0, 1)  # Rearranging shape

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.spline_scaler is not None else 1.0)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
                                 self.scaled_spline_weight.view(self.out_features, -1))

        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                    torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step +
                    x_sorted[0] - margin)

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1), grid,
             grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1)], dim=0)

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                    regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy)



class HGPN(nn.Module):

    def __init__(self, args, adj):
        super(HGPN, self).__init__()
        self.args = args
        self.model_args = args['HGPN']

        self.adj = torch.from_numpy(adj).float().to(args.device)
        self.norm_lap = normalized_laplacian(self.adj)
        self.feature_dim = args[args.dynamics].dim

        # Identity Backbone
        self.repr_net1 = nn.Sequential(
            nn.Linear(self.model_args.n_dim, self.model_args.ag_hid_dim),
            nn.Tanh(),
            nn.Linear(self.model_args.ag_hid_dim, self.model_args.ag_hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.model_args.ag_hid_dim),
        )
        self.repr_net2 = nn.Sequential(
            nn.Linear(self.model_args.n_dim, self.model_args.ag_hid_dim),
            nn.Tanh(),
            nn.Linear(self.model_args.ag_hid_dim, self.model_args.ag_hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.model_args.ag_hid_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

        # State aggregation
        self.agc_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.model_args.ag_hid_dim),
            nn.ReLU(),
            nn.Linear(self.model_args.ag_hid_dim, self.feature_dim),
        )
        self.tanh = nn.Tanh()



        # Backbone Dynamics
        self.BackboneODE = BackboneODE(
            lookback=args.lookback,  # 时间回顾长度
            feature_dim=self.feature_dim,  # 特征维度
            ode_hid_dim=self.model_args.ode_hid_dim,  # ODE 隐藏层维度
            method=self.model_args.method,  # ODE 方法 ('rk4' 等)
            ffcm_dim=self.model_args.ffcm_dim,  # FFCM 维度
            d_model=self.model_args.d_model,  # 模型隐藏维度 (注意力机制)
            n_heads=self.model_args.n_heads,  # 注意力头数
            window_size=self.model_args.window_size,  # 时间窗口大小
            dropout=self.model_args.dropout  # Dropout 比例
        )

        # K-means
        self.cluster_idx, self.cluster_centers = self._kmeans(adj, self.model_args.k, self.model_args.log)

        # Refine
        self.refiners = nn.ModuleList(
            [Refiner(args.lookback, args.horizon, self.feature_dim, self.model_args.sr_hid_dim) for _ in
             range(self.model_args.k)])

        # Device
        self.to(args.device)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Init hyperbolic embedding
        self.node_embedding, angular = self._init_poincare()
        self.supernode_embedding, backbone, assignment_matrix = self._init_super_node(angular)
        draw_embedding(self.adj, self.node_embedding, f'{self.args.log_dir}/HE/init_node_poincare.png')
        draw_embedding(backbone, self.supernode_embedding, f'{self.args.log_dir}/HE/init_supernode_poincare.png')

        if self.model_args.prior_init:
            # Pretrain for Identity Backbone
            self._pretrain_identity_backbone(assignment_matrix)
            draw_embedding(self.backbone, self.supernode_embedding,
                           f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/pretrain_supernode_poincare.png')

    def _init_poincare(self):
        print('Initializing poincare embedding...')

        _, angular, radius = HyperbolicEmbedding(self.args).fit_transform()
        radius = radius / radius.max()  # normalize radius to [0, 1]

        x, y = radius * np.cos(angular), radius * np.sin(angular)
        poincare_embedding = torch.from_numpy(np.stack([x, y], axis=1)).float().to(self.args.device)

        print('Done.')
        return self._check_norm(poincare_embedding), angular

    def _init_super_node(self, angular):
        print('Initializing super node embedding...')
        num = int(self.model_args.ratio * self.args.node_num)

        # init super node embedding by angular
        idx = np.argsort(angular)
        assignment_matrix = torch.zeros(num, self.args.node_num).to(self.args.device)
        size = int(1 / self.model_args.ratio)
        for i in range(num):
            assignment_matrix[i, idx[i * size:(i + 1) * size]] = 1

        degree = self.adj.sum(axis=1, keepdims=True)
        super_node_embedding = assignment_matrix @ (self.node_embedding * degree) / (assignment_matrix @ degree).sum(
            dim=-1, keepdim=True)
        super_node_embedding = nn.Parameter(self._check_norm(super_node_embedding))
        backbone = assignment_matrix @ self.adj @ assignment_matrix.T

        print('Done.')
        return super_node_embedding, backbone, assignment_matrix

    def _pretrain_identity_backbone(self, prior_assignment_matrix):

        optimizer = torch.optim.Adam(
            [
                {'params': self.repr_net1.parameters(), 'lr': self.args.lr},
                {'params': self.repr_net2.parameters(), 'lr': self.args.lr},
                {'params': self.agc_mlp.parameters(), 'lr': self.args.lr},
            ],
            lr=self.args.lr)
        loss_fn = nn.L1Loss()

        for epoch in range(self.model_args.pretrain_epoch):
            # 1. map to euclidean space from poincare space
            node_euclidean_embedding = PoincareManifold.log_map_zero(self.node_embedding)
            supernode_euclidean_embedding = PoincareManifold.log_map_zero(self.supernode_embedding)
            # 2. topology-aware representation
            node_repr = self.repr_net1(node_euclidean_embedding)
            supernode_repr = self.repr_net2(supernode_euclidean_embedding)
            # 3. assignment matrix
            assignment_prob = self.softmax(supernode_repr @ node_repr.T)
            assignment_matrix = assignment_prob
            # 4. loss
            loss = loss_fn(assignment_matrix, prior_assignment_matrix)
            # 5. update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'\rPretrain identity backbone[{epoch}]: {loss.item():.4f}', end='')

        print()
        del optimizer, loss_fn

    @property
    def assignment_matrix(self):
        # 1. map to euclidean space from poincare space
        node_euclidean_embedding = PoincareManifold.log_map_zero(self.node_embedding)
        supernode_euclidean_embedding = PoincareManifold.log_map_zero(self.supernode_embedding)
        # 2. topology-aware representation
        node_repr = self.repr_net1(node_euclidean_embedding)
        supernode_repr = self.repr_net2(supernode_euclidean_embedding)
        # 3. assignment matrix
        assignment_prob = self.softmax(supernode_repr @ node_repr.T)
        assignment_matrix = assignment_prob
        return assignment_matrix

    @property
    def backbone(self):
        assignment_matrix = self.assignment_matrix
        idx = torch.argmax(assignment_matrix, dim=0)
        assignment_matrix = torch.zeros_like(assignment_matrix, device=self.args.device)
        assignment_matrix[idx, torch.arange(idx.shape[0])] = 1
        backbone = assignment_matrix @ self.adj @ assignment_matrix.T
        backbone[backbone > 0] = 1
        return backbone

    def _update_supernode_embedding(self, lr):
        # Update supernode embedding by backbone
        euclidean_grad = self.supernode_embedding.grad
        poincare_grad = PoincareManifold.poincare_grad(euclidean_grad, self.supernode_embedding)
        self.supernode_embedding.data -= lr * poincare_grad
        self.supernode_embedding.data = self._check_norm(self.supernode_embedding.data)
        self.supernode_embedding.grad.zero_()

    def _check_norm(self, embedding, eps=1e-5):
        norm = torch.norm(embedding, dim=-1)

        # Keep the norm of embedding less than 1
        idx = norm > 1
        if idx.sum() > 0:
            embedding[idx] = embedding[idx] / norm[idx].unsqueeze(-1) - eps
        return embedding

    def _kmeans(self, adj, k, log=True):
        assert k >= 1, "k must be greater than 1"

        degree = adj.sum(axis=1)
        if log:
            log_degree = np.log(degree)

        model = cluster.KMeans(n_clusters=k, n_init='auto', max_iter=1000, random_state=0)
        model.fit(np.array(log_degree).reshape(-1, 1))
        labels = model.labels_
        log_centers = model.cluster_centers_

        if log:
            centers = np.exp(log_centers)

        cluster_ids = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            cluster_ids[label].append(i)

        return cluster_ids, centers

    def forward(self, tspan, X, isolate=False):
        # X: (batch_size, lookback, node_num, feature_dim)

        ###################
        # Identity Backbone
        ###################
        # 1. map to euclidean space from poincare space
        node_euclidean_embedding = PoincareManifold.log_map_zero(self.node_embedding)
        supernode_euclidean_embedding = PoincareManifold.log_map_zero(self.supernode_embedding)
        # 2. topology-aware representation
        node_repr = self.repr_net1(node_euclidean_embedding)
        supernode_repr = self.repr_net2(supernode_euclidean_embedding)
        # 3. assignment matrix
        assignment_prob = self.softmax(supernode_repr @ node_repr.T)
        assignment_matrix = assignment_prob
        # 4. backbone
        backbone = assignment_matrix @ self.adj @ assignment_matrix.T

        ###################
        # State aggregation
        ###################
        # 1. dynamics-aware representation
        agc_repr = self.tanh(self.agc_mlp(self.norm_lap @ X))
        # 2. state aggregation
        X_supernode = assignment_matrix @ agc_repr  # batch_size, lookback, supernode_num, feature_dim

        ###################
        # Backbone Dynamics
        ###################
        # 1. predict supernode trajectory by graph neural ode
        Y_supernode = self.BackboneODE(tspan, X_supernode, backbone)  # batch_size, horizon, supernode_num, feature_dim
        # 2. copy supernode trajectory to original nodes
        Y_coarse = assignment_matrix.T @ Y_supernode  # batch_size, horizon, node_num, feature_dim

        ###################
        # Refine
        ###################
        Y_refine = torch.zeros_like(Y_coarse)
        if isolate:
            Y_coarse = Y_coarse.detach()

        for k in range(len(self.refiners)):
            cluster_X = X[:, :, self.cluster_idx[k]]
            cluster_Y_coarse = Y_coarse[:, :, self.cluster_idx[k]]

            if len(self.cluster_idx[k]) == 0:
                continue
            else:
                Y_refine[:, :, self.cluster_idx[k]] = self.refiners[k](cluster_X, cluster_Y_coarse)

        return assignment_matrix, Y_refine, Y_supernode, (Y_coarse, X, X_supernode)

    def _agc_state(self, X, assignment_matrix):
        agc_repr = self.tanh(self.agc_mlp(self.norm_lap @ X))
        X_supernode = assignment_matrix @ agc_repr  # batch_size, lookback, supernode_num, feature_dim
        return X_supernode

    def _rg_loss(self, y_rg, Y, assignment_matrix, dim=None):

        # Averaging Y by RG mapping M
        with torch.no_grad():
            Y_supernode = self._agc_state(Y, assignment_matrix)

        # MSE Loss
        if dim is None:
            rg_loss = torch.mean((y_rg - Y_supernode) ** 2)
        else:
            rg_loss = torch.mean((y_rg - Y_supernode) ** 2, dim=dim)

        return rg_loss, Y_supernode

    def _onehot_loss(self, assignment_matrix):
        entropy = -torch.sum(assignment_matrix * torch.log2(assignment_matrix + 1e-5), dim=0)
        onehot_loss = torch.mean(entropy)
        return onehot_loss

    def _uniform_loss(self, assignment_matrix):
        supernode_strength = torch.sum(assignment_matrix, dim=1)
        prob = supernode_strength / torch.sum(supernode_strength)
        entropy = -torch.sum(prob * torch.log2(prob + 1e-5), dim=0)
        uniform_loss = -torch.mean(entropy)  # maximize entropy
        return uniform_loss

    def _recons_loss(self, assignment_matrix, adj):
        surrogate_adj = assignment_matrix.T @ assignment_matrix
        recons_loss = torch.norm(adj - surrogate_adj, p='fro')
        return recons_loss

    def _refine_loss(self, y_refine, Y, dim=None):

        # MSE Loss
        if dim is None:
            refine_loss = torch.mean((y_refine - Y) ** 2)
        else:
            refine_loss = torch.mean((y_refine - Y) ** 2, dim=dim)

        return refine_loss, Y

    def fit(self, train_dataloader, val_dataloader):

        # if os.path.exists(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model.pt'):
        #     print('Model exists, skip training')
        #     return
        # else:
        #     os.makedirs(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}', exist_ok=True)
        #     print(f'Training {self.args.model} model')

        optimizer = torch.optim.Adam(
            [
                {'params': self.repr_net1.parameters(), 'lr': self.args.lr},
                {'params': self.repr_net2.parameters(), 'lr': self.args.lr},
                {'params': self.agc_mlp.parameters(), 'lr': self.args.lr},
                {'params': self.BackboneODE.parameters(), 'lr': self.args.lr},
                {'params': self.refiners.parameters(), 'lr': self.args.lr},
                {'params': self.supernode_embedding, 'lr': self.args.lr},
            ]
            , lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=self.args.lr_step, gamma=self.args.lr_decay)

        dt = self.args[self.args.dynamics].dt
        start_t = (self.args.lookback) * dt
        end_t = (self.args.lookback + self.args.horizon - 1) * dt
        tspan = torch.linspace(start_t, end_t, self.args.horizon).to(self.args.device)

        train_loss_list, val_loss_list = [], []
        for epoch in range(1, self.args.max_epoch + 1):
            train_loss = 0.0
            self.train()
            for i, (X, Y) in enumerate(train_dataloader):
                assignment_matrix, y_refine, y_rg, _ = self(tspan, X)
                rg_loss, _ = self._rg_loss(y_rg, Y, assignment_matrix)
                refine_loss, _ = self._refine_loss(y_refine, Y)
                onehot_loss = self._onehot_loss(assignment_matrix)
                uniform_loss = self._uniform_loss(assignment_matrix)
                recons_loss = self._recons_loss(assignment_matrix, self.adj)
                loss = refine_loss + rg_loss + onehot_loss + recons_loss + uniform_loss

                optimizer.zero_grad()
                loss.backward()
                self.supernode_embedding.grad = PoincareManifold.poincare_grad(self.supernode_embedding.grad,
                                                                               self.supernode_embedding)  # rescale euclidean grad to poincare grad
                optimizer.step()

                with torch.no_grad():
                    self.supernode_embedding.data = self._check_norm(self.supernode_embedding.data)

                train_loss += loss.item()
            print(
                f'\rEpoch[{epoch}/{self.args.max_epoch}] train backbone: {rg_loss.item():.4f}, refine: {refine_loss.item():.4f}, onehot: {onehot_loss.item():.4f}, recons: {recons_loss.item():.4f}, uniform: {uniform_loss.item():.4f}',
                end='')
            train_loss_list.append([epoch, train_loss / len(train_dataloader)])

            scheduler.step()
            if epoch % self.args.val_interval == 0:
                self.eval()
                val_loss = 0
                for i, (X, Y) in enumerate(val_dataloader):
                    assignment_matrix, y_refine, y_rg, info = self(tspan,
                                                                   X)  # info: (Y_coarse, X_reindex, Y_coarse, X_rg, kappa_reindex)
                    rg_loss, Y_coarse = self._rg_loss(y_rg, Y, assignment_matrix)
                    refine_loss, Y_reindex = self._refine_loss(y_refine, Y)
                    onehot_loss = self._onehot_loss(assignment_matrix)
                    uniform_loss = self._uniform_loss(assignment_matrix)
                    recons_loss = self._recons_loss(assignment_matrix, self.adj)
                    loss = refine_loss
                    val_loss += loss.item()

                    if i == 0:
                        os.makedirs(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}',
                                    exist_ok=True)
                        drawTraj(y_rg[:, :, :100], Y_coarse[:, :, :100], 'pred', 'true', dim=0,
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/rg_pred.png')
                        drawTraj(info[2][:, :, :10], Y_coarse[:, :12, :10], 'rg_x', 'rg_y', dim=0,
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/rg_traj.png')
                        drawTraj(info[1][:, :, :20], info[2][:, :, :10], 'x', 'x_rg', dim=0,
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/x_rg.png')
                        drawTraj(y_refine[:, :, :100], info[0][:, :, :100], 'refined', 'coarse', dim=0,
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine.png')
                        drawTraj(y_refine[:, :, :100], Y_reindex[:, :, :100], 'pred', 'true', dim=0,
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/result.png')
                        drawTraj(Y_reindex[:, :, :100], Y_coarse[:, :, :50], 'Y', 'Y_coarse', dim=0,
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/y_rg.png')

                        # Draw the backbone
                        idx = torch.argmax(assignment_matrix, dim=0)
                        assignment_matrix = torch.zeros_like(assignment_matrix, device=self.args.device)
                        assignment_matrix[idx, torch.arange(idx.shape[0])] = 1
                        backbone = assignment_matrix @ self.adj @ assignment_matrix.T
                        # backbone[backbone > 0] = 1
                        draw_embedding(backbone, self.supernode_embedding,
                                       f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/supernode_poincare.png')

                        # Assignment distribution
                        count = torch.sum(assignment_matrix, dim=1)
                        valid_num = len(count[count > 0])

                print(
                    f'\nEpoch[{epoch}/{self.args.max_epoch}] val backbone: {rg_loss.item():.4f}, refine: {refine_loss.item():.4f}, onehot: {onehot_loss.item():.4f}, recons: {recons_loss.item():.4f}, uniform: {uniform_loss.item():.4f} | assignment: {valid_num}/{self.supernode_embedding.shape[0]}')
                val_loss_list.append([epoch, val_loss / len(val_dataloader)])

                # Save model
                torch.save(self.state_dict(),
                           f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model_{epoch}.pt')

        # Draw loss curve
        train_loss_list = np.array(train_loss_list)
        val_loss_list = np.array(val_loss_list)
        plt.figure(figsize=(5, 4))
        plt.plot(train_loss_list[:, 0], train_loss_list[:, 1], label='train')
        plt.plot(val_loss_list[:, 0], val_loss_list[:, 1], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/loss.png', dpi=300)

        # Save model
        torch.save(self.state_dict(), f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model.pt')

        # Fine-tune refiner
        self.refine(train_dataloader, val_dataloader)

        # Release memory
        del train_dataloader, val_dataloader, optimizer, scheduler

    def refine(self, train_dataloader, val_dataloader):

        dt = self.args[self.args.dynamics].dt
        start_t = (self.args.lookback) * dt
        end_t = (self.args.lookback + self.args.horizon - 1) * dt
        tspan = torch.linspace(start_t, end_t, self.args.horizon).to(self.args.device)

        optimizer = torch.optim.Adam(self.refiners.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=self.args.lr_step, gamma=self.args.lr_decay)

        for epoch in range(1, 10 + 1):
            train_loss = 0.0
            self.train()
            for i, (X, Y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                _, y_refine, y_rg, _ = self(tspan, X, isolate=True)
                refine_loss, _ = self._refine_loss(y_refine, Y)
                loss = refine_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f'\rEpoch[{epoch}] train refine loss: {refine_loss.item():.4f}', end='')

            scheduler.step()
            if epoch % self.args.val_interval == 0:
                self.eval()
                val_loss = 0
                for i, (X, Y) in enumerate(val_dataloader):
                    _, y_refine, y_rg, info = self(tspan, X,
                                                   isolate=True)  # info: (Y_coarse, X_reindex, Y_coarse, X_rg, kappa_reindex)
                    refine_loss, Y_reindex = self._refine_loss(y_refine, Y)
                    loss = refine_loss
                    val_loss += loss.item()
                    if i == 0:
                        os.makedirs(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine',
                                    exist_ok=True)
                        drawTraj(info[1][:, :, :20], info[2][:, :, :10], 'x', 'x_rg',
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine/x_rg.png')
                        drawTraj(y_refine[:, :, :100], info[0][:, :, :100], 'refined', 'coarse',
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine/refine.png')
                        drawTraj(y_refine[:, :, :100], Y_reindex[:, :, :100], 'pred', 'true',
                                 out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine/result.png')
                print(f'\nEpoch[{epoch}/10] val refine loss: {refine_loss.item():.4f}')

        # Save model
        torch.save(self.state_dict(), f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model_refine.pt')

    def test(self, test_dataloader):

        # Load model
        try:
            self.load_state_dict(
                torch.load(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model_refine.pt'))
        except:
            self.load_state_dict(torch.load(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model.pt'))
        self.to(self.args.device)

        dt = self.args[self.args.dynamics].dt
        start_t = (self.args.lookback) * dt
        end_t = (self.args.lookback + self.args.horizon - 1) * dt
        tspan = torch.linspace(start_t, end_t, self.args.horizon).to(self.args.device)

        # Test
        self.eval()
        print('1Testing...')
        ground_truth = np.zeros(
            (len(test_dataloader), self.args.batch_size, self.args.horizon, self.args.node_num, self.feature_dim))
        predict = np.zeros(
            (len(test_dataloader), self.args.batch_size, self.args.horizon, self.args.node_num, self.feature_dim))
        for i, (X, Y) in enumerate(test_dataloader):
            assignment_matrix, y_refine, y_rg, info = self(tspan, X)

            if i == len(test_dataloader) - 1:
                Y_coarse = self._agc_state(Y, assignment_matrix)
                os.makedirs(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test', exist_ok=True)
                drawTraj(y_rg[:, :, :100], Y_coarse[:, :, :100], 'pred', 'true',
                         out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/rg_pred.png', num=3)
                drawTraj(info[2][:, :, :10], Y_coarse[:, :12, :10], 'rg_x', 'rg_y', dim=0,
                         out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/rg_traj.png')
                drawTraj(info[1][:, :, :20], info[2][:, :, :10], 'x', 'x_rg', dim=0,
                         out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/x_rg.png')
                drawTraj(y_refine[:, :, :200], info[0][:, :, :200], 'refined', 'coarse',
                         out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/refine.png')
                drawTraj(y_refine[:, :, :200], Y[:, :, :200], 'pred', 'true',
                         out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/result.png', num=3)
                drawTraj(Y[:, :, :200], Y_coarse[:, :, :100], 'Y', 'Y_coarse',
                         out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/y_rg.png')

            ground_truth[i, :, :, :, :] = Y.cpu().detach().numpy()
            predict[i, :, :, :, :] = y_refine.cpu().detach().numpy()

        backbone_pred = y_rg.cpu().detach().numpy()
        backbone_true = Y_coarse.cpu().detach().numpy()

        # Draw the backbone
        idx = torch.argmax(assignment_matrix, dim=-1)
        assignment_matrix = torch.zeros_like(assignment_matrix)
        assignment_matrix[torch.arange(idx.shape[0]), idx] = 1
        backbone = assignment_matrix @ self.adj @ assignment_matrix.T
        # backbone[backbone > 0] = 1
        draw_embedding(backbone, self.supernode_embedding,
                       f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/supernode_poincare.png')

        # 在绘制图像之前，计算 MSE
        # 在绘制图像之前，计算 MSE
        mse_pred_true = mean_squared_error(y_refine.cpu().detach().numpy().flatten(),
                                           Y.cpu().detach().numpy().flatten())

        print(f'MSE between predictions and ground truth: {mse_pred_true}')
        r2_pred_true = r2_score(y_refine.cpu().detach().numpy().flatten(),
                                Y.cpu().detach().numpy().flatten())
        print(f'R² between predictions and ground truth: {r2_pred_true}')
        # Example: Show the result of prediction vs. ground truth
        # 计算 MAE
        mae_pred_true = mean_absolute_error(y_refine.cpu().detach().numpy().flatten(),
                                            Y.cpu().detach().numpy().flatten())
        print(f'MAE between predictions and ground truth: {mae_pred_true}')

        # 计算真实值的均值
        mean_true = np.mean(Y.cpu().detach().numpy())
        print(f'Mean of ground truth: {mean_true}')

        print('555555555')

        # Save result
        time_cost = 0.0
        ground_truth = ground_truth.reshape(-1, self.args.horizon, self.args.node_num, self.feature_dim)
        predict = predict.reshape(-1, self.args.horizon, self.args.node_num, self.feature_dim)
        np.savez(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/backbone_result.npz',
                 backbone_pred=backbone_pred, backbone_true=backbone_true)
        np.savez(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/result.npz', ground_truth=ground_truth,
                 predict=predict, time_cost=time_cost)

