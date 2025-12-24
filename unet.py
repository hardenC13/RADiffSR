import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from torchvision.models import vgg19
from torch.autograd import Variable
from math import exp
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        #self.up = nn.Upsample(scale_factor=2, mode="bicubic")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.invertedblock = invertedBlock(dim, dim)
    def forward(self, x):
        x = self.invertedblock(x)
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)
        self.invertedblock = invertedBlock(dim,dim)
    def forward(self, x):
        x = self.invertedblock(x)
        return self.conv(x)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            #Mish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class RAEBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.naf = NAFBlock(dim_out)



    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)

        h = self.naf(h)

        return h + self.res_conv(x)
##########################  通道注意力和空间注意力  ############################
class CLAM(nn.Module):
    def __init__(self, in_planes, ratio=16, pool_mode='Avg|Max'):
        super(CLAM, self).__init__()
        self.pool_mode = pool_mode
        if pool_mode.find('Avg') != -1:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool_mode.find('Max') != -1:
            self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        # self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        if self.pool_mode == 'Avg':
            out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        elif self.pool_mode == 'Max':
            out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        elif self.pool_mode == 'Avg|Max':
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
        out = self.sigmoid(out) * x
        return out

class SLAM(nn.Module):
    def __init__(self, kernel_size=7, pool_mode='Avg|Max'):
        super(SLAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.pool_mode = pool_mode
        if pool_mode == 'Avg|Max':
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.pool_mode == 'Avg':
            out = torch.mean(x, dim=1, keepdim=True)
        elif self.pool_mode == 'Max':
            out, _ = torch.max(x, dim=1, keepdim=True)
        elif self.pool_mode == 'Avg|Max':
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out)) * x
        return out

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = RAEBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        self.conv = nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=True)
        if with_attn:
            self.ca = CLAM(dim_out, pool_mode='Avg|Max')
            self.sa = SLAM(kernel_size=7, pool_mode='Avg|Max')
            # self.nlsa = RDB(dim_out, dim_out, 3)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.ca(x)
            x = self.sa(x)
            # y = self.nlsa(x)
            # x = x + y
        return x



class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 4),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=256
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                #Mish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = False  # True
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = False  # True
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                # 获取编码器对应层的通道数
                enc_channels = feat_channels[-1]
                dec_channels = channel_mult
                ups.append(UpBlockWithCSDG(
                    pre_channel+feat_channels.pop(), channel_mult,enc_channels,dec_channels, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, x_cpem, use_cpem):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        is_first_loop = True
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
                if is_first_loop and use_cpem:
                    x = x + x_cpem
                    is_first_loop = False
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        for layer in self.ups:
            if isinstance(layer, UpBlockWithCSDG):
                skip = feats.pop()
                x = layer(x, skip, t)
            else:
                x = layer(x)

        return self.final_conv(x)

# =============================================================================NAFblock==========================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
# =============================================================================invertedBlock==========================
# 先加在下采样前 X
# 上下采样都加
class invertedBlock(nn.Module):
    def __init__(self, in_channel, out_channel,ratio=2):
        super(invertedBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.GELU()
        ## 7*7卷积，并行3*3卷积
        self.conv1 = nn.Conv2d(internal_channel, internal_channel, 7, 1, 3, groups=in_channel,bias=False)

        self.convFFN = FFN(in_channels=in_channel, out_channels=in_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.pw1 = nn.Conv2d(in_channels=in_channel, out_channels=internal_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channel, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
    def hifi(self,x):

        x1=self.pw1(x)
        x1=self.relu(x1)
        x1=self.conv1(x1)
        x1=self.relu(x1)
        x1=self.pw2(x1)
        x1=self.relu(x1)
        # x2 = self.conv2(x)
        x3 = x1+x

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.layer_norm(x3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = self.convFFN(x3)

        return x4

    def forward(self, x):
        return self.hifi(x)+x
class FFN(nn.Module):

    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()

        internal_channels = in_channels * expend_ratio
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        x1 = self.pw1(x)
        x2 = self.nonlinear(x1)
        x3 = self.pw2(x2)
        x4 = self.nonlinear(x3)
        return x4 + x
# =============================================================================CS-DG==========================
class CrossScaleDynamicGating(nn.Module):
    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        # 通道注意力（处理拼接后的特征）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(enc_channels + dec_channels, (enc_channels + dec_channels) // 4, 1),
            nn.ReLU(),
            nn.Conv2d((enc_channels + dec_channels) // 4, enc_channels + dec_channels, 1),
            nn.Sigmoid()
        )
        # 通道压缩层（将 enc+dec 压缩至 dec_channels）
        self.channel_reduce = nn.Conv2d(enc_channels + dec_channels, dec_channels, 1)
        # 空间注意力（作用于压缩后的 dec_channels）
        self.spatial_att = nn.Sequential(nn.Conv2d(dec_channels, 1, 7, padding=3),nn.Sigmoid())
    def forward(self, x_enc, x_dec):
        # 确保编码器和解码器特征形状匹配
        assert x_enc.shape[1] == self.enc_channels, f"Encoder channel mismatch: {x_enc.shape[1]} vs {self.enc_channels}"
        assert x_dec.shape[1] == self.dec_channels, f"Decoder channel mismatch: {x_dec.shape[1]} vs {self.dec_channels}"
        # 特征拼接
        x_cat = torch.cat([x_enc, x_dec], dim=1)  # [B, enc+dec, H, W]
        # 通道注意力加权
        channel_weight = self.channel_att(x_cat)  # [B, enc+dec, 1, 1]
        x_weighted = x_cat * channel_weight  # [B, enc+dec, H, W]
        # 通道压缩
        x_reduced = self.channel_reduce(x_weighted)  # [B, dec, H, W]
        # 空间注意力加权
        spatial_weight = self.spatial_att(x_reduced)  # [B, 1, H, W]
        x_out = x_reduced * spatial_weight  # [B, dec, H, W]
        # 残差连接（确保与解码器特征维度一致）
        return x_out + x_dec  # [B, dec, H, W]
class UpBlockWithCSDG(nn.Module):
    def __init__(self, dim, dim_out,enc_channels,dec_channels, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = RAEBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        self.conv = nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=True)
        self.cs_dg = CrossScaleDynamicGating(enc_channels,dec_channels)  # 新增CS-DG模块

        if with_attn:
            self.ca = CLAM(dim_out, pool_mode='Avg|Max')
            self.sa = SLAM(kernel_size=7, pool_mode='Avg|Max')
            # self.nlsa = RDB(dim_out, dim_out, 3)

    def forward(self, x, skip,time_emb):
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x, time_emb)
        x = self.cs_dg(skip, x)
        if(self.with_attn):
            x = self.ca(x)
            x = self.sa(x)
        return x

if __name__ == "__main__":
    enc = torch.randn(1, 256, 32, 32)  # enc_channels=64
    dec = torch.randn(1, 128, 32, 32) # dec_channels=128
    csdg = CrossScaleDynamicGating(256, 128)
    output = csdg(enc, dec)
    assert output.shape == dec.shape, f"Output shape {output.shape} != Decoder shape {dec.shape}"