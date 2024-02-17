# NM: 97.45% BG: 95.18% CL: 87.13%

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from einops import rearrange
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, \
    BasicConv2d, DilatedConv3d, BasicConv3d

class RandomMask(nn.Module):
    def __init__(self, maskRateH, maskRateW, maskRateFrame):
        super(RandomMask, self).__init__()
        self.H = maskRateH
        self.W = maskRateW
        self.F = maskRateFrame

    def forward(self, x):
        n, c, s, h, w = x.size()

        # 划定mask的起始像素点的范围
        lenH = int(h * self.H)
        lenW = int(w * self.W)
        range_h = range(0, int((1 - self.H) * h)+1)
        range_w = range(0, int((1 - self.W) * w)+1)

        # 得到随机像素点
        sample_h = random.sample(range_h, int(n * self.F))
        sample_w = random.sample(range_w, int(n * self.F))

        # 得到x中的随机帧
        sample_frame = random.sample(range(0, n), int(n * self.F))

        # 对x进行mask操作
        for k, i, j in zip(sample_frame, sample_h, sample_w):
            x[k, :, :, i:i+lenH, j:j+lenW] = 0

        return x

class edgeMask(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                 padding=(1, 1, 1), bias=False, **kwargs):
        super(edgeMask, self).__init__()
        self.local = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.max = nn.MaxPool3d(kernel_size=(kernel_size[0], 1, 1), stride=stride, padding=(padding[0], 0, 0))
        self.avg = nn.AvgPool3d(kernel_size=(kernel_size[0], 1, 1), stride=stride, padding=(padding[0], 0, 0))
        self.bn = nn.BatchNorm3d(out_channels)


    def forward(self, x):
        # n, c, s, h, w = x.size()

        edgeMask_max = self.max(x)
        edgeMask_avg = self.avg(x)
        edgeMask_feature = torch.sigmoid(edgeMask_max - edgeMask_avg)
        edgeMask_one = (edgeMask_feature > 0.7).float()
        # edgeMask_zero = 1. - edgeMask_one

        x_edge = x * edgeMask_one
        x_inner = x * (1. - edgeMask_one)
        feature_edge = self.local(x_edge)
        feature_inner = self.local(x_inner)
        feature_local = feature_edge + feature_inner
        feature_local = self.bn(feature_local)
        return feature_local

class GLConv_mask(nn.Module):
    def __init__(self, in_channels, out_channels, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1)
                 , padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv_mask, self).__init__()
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

        self.local_conv3d = edgeMask(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        # n, c, s, h, w = x.size()
        # if s < 3:
        #     repeat = 3 if s == 1 else 2
        #     x = x.repeat(1, 1, repeat, 1, 1)

        gob_feat = self.global_conv3d(x)
        lcl_feat = self.local_conv3d(x)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return self.bn(feat)

class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1) * p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class MSTA(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=(1, 1, 1), kernel_size=(3, 1, 1), stride=(1, 1, 1),
                 padding=(1, 0, 0), bias=False, flag=False, **kwargs):
        super(MSTA, self).__init__()

        self.dilate_conv1 = DilatedConv3d(
            in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias)

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.dilate_conv2 = DilatedConv3d(
            in_channels=out_channels, out_channels=out_channels, dilation=dilation, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias)

        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout(p=0.2)
        self.flag = flag

        if self.flag:
            self.conv1x1x1 = BasicConv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)

    def forward(self, x):
        '''
            x: [n, c0, s, h, w]
            out: [n, c1, s, h, w]
        '''
        residule = x  # [n, c0, s, h, w]

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dilate_conv1(out)

        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dilate_conv2(out)

        if self.flag:
            residule = self.conv1x1x1(x)  # [n, c0->c1, s, h, w]

        out = out + residule

        return out


class MyModel(BaseModel):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, *args, **kargs):
        super(MyModel, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']

        if dataset_name in ['OUMVLP', 'GREW']:
            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )

            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )
            # self.LTA = nn.Sequential(
            #     DilatedConv3d(in_c[0], in_c[0], dilation=(2, 1, 1), kernel_size=(
            #         3, 1, 1), stride=(3, 1, 1), padding=(2, 0, 0)),
            #     nn.LeakyReLU(inplace=True)
            # )

            # self.LTA = nn.Sequential(
            #     MSTA(in_c[0], in_c[0]),
            #     nn.LeakyReLU(inplace=True)
            # )
            self.RM = RandomMask(0.5, 0.5, 0.1)

            self.GLConvA = nn.Sequential(
                GLConv_mask(in_c[0], in_c[1], fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv_mask(in_c[1], in_c[1], fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )

            self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvB = nn.Sequential(
                GLConv_mask(in_c[1], in_c[2], fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv_mask(in_c[2], in_c[2], fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )

            self.GLConvC = nn.Sequential(
                GLConv_mask(in_c[2], in_c[3], fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv_mask(in_c[3], in_c[3], fm_sign=True, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )

            # Mutil-Scale Temporal Feature Aggregation Module
            # self.dlaConv3d_1 = MSTA(in_c[2], in_c[3], dilation=(1, 1, 1), kernel_size=(3, 1, 1), stride=(1, 1, 1),
            #                         padding=(1, 0, 0), flag=True, bias=False)

            self.dlaConv3d_2 = MSTA(in_c[3], in_c[3], dilation=(2, 1, 1), kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                    padding=(2, 0, 0), flag=False, bias=False)

            # self.dlaConv3d_4 = MSTA(in_c[3], in_c[3], dilation=(4, 1, 1), kernel_size=(3, 1, 1), stride=(1, 1, 1),
            #                         padding=(4, 0, 0), flag=False, bias=False)

        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )

            # self.LTA = nn.Sequential(
            #     BasicConv3d(in_c[0], in_c[0], kernel_size=(
            #         3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            #     nn.LeakyReLU(inplace=True)
            # )
            # 加mask的config

            # self.RM = RandomMask(0.5, 0.5, 0.1)
            self.GLConvA = GLConv_mask(
                in_c[0], in_c[1], fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

            self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvB = GLConv_mask(
                in_c[1], in_c[2], fm_sign=True, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

            # self.GLConvC = GLConv_mask(
            #     in_c[2], in_c[2], fm_sign=True, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

            # Mutil-Scale Temporal Feature Aggregation Module
            # self.dlaConv3d_1 = MSTA(in_c[2], in_c[3], dilation=(1, 1, 1), kernel_size=(3, 3, 3), stride=(1, 1, 1),
            #                        padding=(1, 1, 1), flag=True, bias=False)

            self.dlaConv3d_2 = MSTA(in_c[2], in_c[3], dilation=(2, 1, 1), kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                    padding=(2, 0, 0), flag=True, bias=False)

            self.dlaConv3d_4 = MSTA(in_c[3], in_c[3], dilation=(4, 1, 1), kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                   padding=(4, 0, 0), flag=False, bias=False)
            # self.LTA = nn.Sequential(
            #     DilatedConv3d(in_c[0], in_c[0], dilation=(2, 1, 1), kernel_size=(
            #         3, 1, 1), stride=(3, 1, 1), padding=(2, 0, 0)),
            #     nn.LeakyReLU(inplace=True)
            # )

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)  # [n, 1, s, h, w]
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        # 进行RandomMask操作
        # sils = self.RM(sils)

        outs = self.conv3d(sils)  # [n, c0, s1, h ,w]

        # outs = self.LTA(outs)

        outs = self.GLConvA(outs)  # [n, c1, s1, h, w]

        outs = self.MaxPool0(outs)  # spatial pooling: [n, c2, s0, h, w]  ->  [n, c2, s0, h/2, w/2]

        outs = self.GLConvB(outs)  # [n, c2, s1, h, w/2]

        # outs = self.GLConvC(outs)  # [n, c2, s1, h, w/2]

        # Mutil-Scale Temporal Feature Aggregation Module
        outs = self.dlaConv3d_2(outs)
        #
        outs = self.dlaConv3d_4(outs)
        # outs = self.dlaConv3d_1(outs)

        # outs = self.dlaConv3d(outs)

        #outs = torch.cat([feature_S, feature_Tem], dim=3)

        # get finally feature
        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c2, h, w/2]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
