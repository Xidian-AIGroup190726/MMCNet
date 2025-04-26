from einops import rearrange
from MMCNet.pan_mamba import BasicConv2d, PANMamba, MSMamba, FusionMamba
from MMCNet.loss import FocalLoss
from MMCNet.mbcc import MBCC
from torch import nn
import torch


class mynet(nn.Module):
    def __init__(self, dim=64, num_classes=7,
                 use_reconstruct=True, gate_threshold=0.5,
                 use_exchange=True, use_sobel=False,
                 use_mha=True, lam=0.25, learn=True,
                 Ms4_patch_size=16,
                 hidden_expand=4,
                 down_rate=[2, 2, 2],
                 patch_size=[2, 2, 1],
                 ):
        super().__init__()

        self.PAN_embed = BasicConv2d(1, dim, kernel_size=3, stride=1, padding=1)

        insize2 = Ms4_patch_size * 4 // down_rate[0]
        insize3 = insize2 // down_rate[1]
        out_size = insize3 // down_rate[2]
        # out_size = insize4 // down_rate[3]

        self.PAN_1 = PANMamba(use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
                              down_rate=down_rate[0], dim_expand_rate=hidden_expand,
                              patch_size=patch_size[0], in_dim=dim, in_size=Ms4_patch_size * 4)
        self.PAN_2 = PANMamba(use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
                              down_rate=down_rate[1], dim_expand_rate=hidden_expand,
                              patch_size=patch_size[1], in_dim=dim, in_size=insize2)
        self.PAN_3 = PANMamba(use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
                              down_rate=down_rate[2], dim_expand_rate=hidden_expand,
                              patch_size=patch_size[2], in_dim=dim, in_size=insize3)
        # self.PAN_4 = PANMamba(use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
        #                       down_rate=down_rate[3], dim_expand_rate=hidden_expand,
        #                       patch_size=patch_size[3], in_dim=dim, in_size=insize4)

        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.MS_embed = BasicConv2d(4, dim, kernel_size=3, stride=1, padding=1)

        self.MS_1 = MSMamba(use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
                            down_rate=down_rate[0], dim_expand_rate=hidden_expand,
                            patch_size=patch_size[0], in_dim=dim, in_size=Ms4_patch_size * 4)
        self.MS_2 = MSMamba(use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
                            down_rate=down_rate[1], dim_expand_rate=hidden_expand,
                            patch_size=patch_size[1], in_dim=dim, in_size=insize2)
        self.MS_3 = MSMamba(use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
                            down_rate=down_rate[2], dim_expand_rate=hidden_expand,
                            patch_size=patch_size[2], in_dim=dim, in_size=insize3)
        # self.MS_4 = MSMamba(use_reconstruct=use_reconstruct, gate_threshold=gate_threshold,
        #                     down_rate=down_rate[3], dim_expand_rate=hidden_expand,
        #                     patch_size=patch_size[3], in_dim=dim, in_size=insize4)

        self.fusion_mamba_1 = FusionMamba(d_model=dim, expand=1, use_exchange=use_exchange, use_sobel=use_sobel)
        self.fusion_mamba_2 = FusionMamba(d_model=dim, expand=1, use_exchange=use_exchange, use_sobel=use_sobel)
        self.fusion_mamba_3 = FusionMamba(d_model=dim, expand=1, use_exchange=use_exchange, use_sobel=use_sobel)
        # self.fusion_mamba_4 = FusionMamba(d_model=dim, expand=1, use_exchange=use_exchange, use_sobel=use_sobel)
        self.use_mha = use_mha
        if self.use_mha:
            self.outlayer = MBCC(lam=lam, input_dim=dim, num_classes=num_classes, learn=learn)

        else:
            self.outlayer = nn.Sequential(nn.Linear(dim * out_size * out_size, dim * out_size),
                                          nn.SiLU(),
                                          nn.Linear(dim * out_size, dim),
                                          nn.SiLU(),
                                          nn.Linear(dim, num_classes))

        # self.Loss_bce = nn.CrossEntropyLoss()

        self.Focal_loss = FocalLoss(num_classes)
        # 参数初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, pan, ms, label=None):
        pan = self.PAN_embed(pan)
        ms = self.up(ms)
        ms = self.MS_embed(ms)

        pan = self.PAN_1(pan)
        ms = self.MS_1(ms)
        pan_1, ms_1 = self.fusion_mamba_1(pan, ms)

        pan = self.PAN_2(pan + pan_1)
        ms = self.MS_2(ms + ms_1)
        pan_2, ms_2 = self.fusion_mamba_2(pan, ms)

        pan = self.PAN_3(pan + pan_2)
        ms = self.MS_3(ms + ms_2)
        pan_3, ms_3 = self.fusion_mamba_3(pan, ms)

        # pan = self.PAN_4(pan + pan_3)
        # ms = self.MS_4(ms + ms_3)
        # pan_4, ms_4 = self.fusion_mamba_4(pan, ms)

        if self.use_mha:
            out_1 = self.outlayer(pan_1 + ms_1)
            out_2 = self.outlayer(pan_2 + ms_2)
            out_3 = self.outlayer(pan_3 + ms_3)
            out = out_1 + out_2 + out_3
            # out = self.outlayer(pan_3 + ms_3)
            #
            # out_4 = self.outlayer(pan_4 + ms_4)

            # out = self.outlayer(pan_4 + ms_4)
        else:
            f_out = pan_3 + ms_3
            f_out = f_out.contiguous().view(f_out.shape[0], -1)
            out = self.outlayer(f_out)
        if self.training:

            bce_loss = self.Focal_loss(out, label)

            return bce_loss

        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


if __name__ == '__main__':

    pan = torch.rand(2, 1, 64, 64).cuda()
    ms = torch.rand(2, 4, 16, 16).cuda()
    # mshpan = torch.randn(2, 1, 64, 64)
    model = mynet(dim=64, num_classes=12, lam=0.25, gate_threshold=0.5).cuda()
    from thop import profile

    flops, params = profile(model, (pan, ms,))
    # print('flops: ', flops, 'params: ', params)
    print('Flops:', "%.2fM" % (flops / 1e6), 'Params:', "%.2fM" % (params / 1e6))

