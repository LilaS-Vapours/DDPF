import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import numpy as np
import math
from einops import rearrange


def get_syncbn():
    # return nn.BatchNorm2d
    return nn.SyncBatchNorm


class dec_deeplabv3_plus(nn.Module):
    def __init__(
            self,
            in_planes,
            num_classes=19,
            inner_planes=256,
            sync_bn=False,
            dilations=(12, 24, 36),
            low_conv_planes=48,
    ):
        super(dec_deeplabv3_plus, self).__init__()
        self.is_corr = True
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, low_conv_planes, kernel_size=1),
            norm_layer(low_conv_planes),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)

        # self.head = nn.Sequential(
        #     nn.Conv2d(self.aspp.get_outplanes(), 256, 1, bias=False),
        #     norm_layer(256),
        #     nn.ReLU(inplace=True),
        # )

        self.classifier = nn.Sequential(
            nn.Conv2d(inner_planes + int(low_conv_planes), 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # self.aspp_decode = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)

        # enc_channels = [256, 512, 1024, 2048]
        #
        # self.head1 = ASPPModule(enc_channels[1], dilations)
        # self.head2 = ASPPModule(enc_channels[2], dilations)
        # self.head3 = ASPPModule(enc_channels[3], dilations)
        #
        #
        # self.fuse1 = nn.Sequential(nn.Conv2d(enc_channels[1] // 8 + 48, 256, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(True),
        #                            nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(True))
        # self.fuse2 = nn.Sequential(nn.Conv2d(enc_channels[2] // 8 + 48, 256, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(True),
        #                            nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(True))
        # self.fuse3 = nn.Sequential(nn.Conv2d(enc_channels[3] // 8 + 48, 256, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(True),
        #                            nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(True))
        #
        # self.classifier_fp = nn.Conv2d(256, num_classes, 1, bias=True)

        if self.is_corr:
            self.corr = Corr(nclass=num_classes)
            self.proj = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )

    def _decode(self, c1, c4):
        c4 = self.aspp(c4)
        # c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.low_conv(c1)

        feature = torch.cat([c1, c4], dim=1)
        out = self.classifier(feature)

        return out

    def _decode_ms(self, c1, c2, c3, c4):
        c2 = self.aspp(c2, inner_planes=64)
        # c2 = self.head(c2)
        c2 = F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c3 = self.aspp(c3, inner_planes=128)
        # c3 = self.head2(c3)
        c3 = F.interpolate(c3, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c4 = self.aspp(c4)
        # c4 = self.head3(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.low_conv(c1)  # c1,c2,c3,c4:  48,64,128,256

        feature1 = torch.cat([c1, c2], dim=1)
        # feature1 = self.classifier(feature1,inner_planes=64)
        # out1 = self.classifier3(feature1)
        feature2 = torch.cat([feature1, c3], dim=1)
        # feature2 = self.classifier(feature2,inner_planes=128)
        # out2 = self.classifier3(feature2)
        feature3 = torch.cat([feature2, c4], dim=1)
        feature = self.classifier(feature3, inner_planes=256 + 128 + 64)
        # out3 = self.classifier3(feature3)

        return feature

    def forward(self, x, need_fp = False): #need_fp=False,use_corr=False
        # dict_return = {}
        x1, x2, x3, x4 = x  # 256,512,1024,2048
        # print(x1.size(),x2.size(),x3.size(),x4.size())
        # print(low_feat.size())
        h, w = x1.size()[-2:]

        if need_fp:
            outs = self._decode(torch.cat((x1, nn.Dropout2d(0.5)(x1))),
                                torch.cat((x4, nn.Dropout2d(0.5)(x4))))

            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            proj_feats = self.proj(x4)
            corr_out_map, corr_out, corr_out_dict = self.corr(proj_feats, out)
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)

            return out, out_fp, corr_out

        low_feat = self.low_conv(x1)
        aspp_out = self.aspp(x4)
        # print(aspp_out.size())
        # aspp_out = self.head(aspp_out)
        out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        out = torch.cat((low_feat, out), dim=1)
        out = self.classifier(out)

        # if use_corr:
        #     proj_feats = self.proj(x4)
        #     corr_out_dict = self.corr(proj_feats, out)
        #     dict_return['corr_map'] = corr_out_dict['corr_map']
        #     corr_out = corr_out_dict['out']
        #     corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
        #     dict_return['corr_out'] = corr_out
        #     dict_return['out'] = out
        #     return dict_return

        return out


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res


class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
            self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)
    ):
        super(ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.out_planes = (len(dilations) + 2) * inner_planes

        self.head = nn.Sequential(
            nn.Conv2d(self.out_planes, inner_planes, 1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        aspp_out = self.head(aspp_out)
        return aspp_out


class Corr(nn.Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):
        dict_return = {}
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        h_out, w_out = out.shape[2], out.shape[3]
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)

        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = rearrange(out, 'n c h w -> n c (h w)')
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        corr_map_sample = self.sample(corr_map.detach(), h_in, w_in)

        corr_map_out = self.normalize_corr_map(corr_map_sample, h_in, w_in, h_out, w_out)
        corr_out = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)
        # dict_return['corr_map'] = self.normalize_corr_map(corr_map_sample, h_in, w_in, h_out, w_out)
        # dict_return['out'] = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)

        return corr_map_out,corr_out,dict_return

    def sample(self, corr_map, h_in, w_in):
        index = torch.randint(0, h_in * w_in - 1, [128])
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample

    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m, hw = corr_map.shape
        corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

        corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
        temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = (temp_map > 0.5)
        norm_corr_map = rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        return norm_corr_map
