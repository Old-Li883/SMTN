from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
from torch.nn import MultiheadAttention, Dropout
import torch.nn.functional as F
from torch import matmul,cat
from pysot.core.config import cfg
from pysot.models.utile.loss import select_cross_entropy_loss, IOULoss, DISCLE
from pysot.models.backbone.backbone import backbone

from pysot.models.utile.utile import SMTN
import matplotlib.pyplot as plt

import numpy as np
import cv2


class _NonLocalBlock2D(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(_NonLocalBlock2D, self).__init__()
        self.dimension = 2
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class CrossLocal(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(CrossLocal, self).__init__()

        self.in_channels = in_channels
        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), bn(self.in_channels))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, main_feature, cross_feature):
        batch_size = main_feature.size(0)
        main_size = main_feature.size(2)
        cross_size = cross_feature.size(2)

        x = self.g(cross_feature).view(batch_size, self.inter_channels, -1)
        x = x.permute(0, 2, 1)

        y = self.theta(cross_feature).view(batch_size, self.inter_channels, -1)

        z = F.interpolate(main_feature,
                            cross_size,
                            mode='bilinear',
                            align_corners=False)

        z = self.phi(z).view(batch_size, self.inter_channels, -1)
        z = z.permute(0, 2, 1)

        f = t.matmul(x, y)
        f_div_C = F.softmax(f, dim=-1)

        output = t.matmul(f_div_C, z)
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(batch_size, self.inter_channels,
                             *cross_feature.size()[2:])

        output = self.W(output)
        output = F.interpolate(output,
                                 main_size,
                                 mode='bilinear',
                                 align_corners=False)
        output += main_feature
        return output

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out

class featureFusion(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,activation='relu'):
        super(featureFusion,self).__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)


    def forward(self, q, k, v):
        src = self.self_attn1(q, k, v)[0]
        srcs1 = v + self.dropout1(src)
        srcs1 = self.norm1(srcs1)

        src2 = self.self_attn2(srcs1,srcs1,srcs1)[0]
        srcs2 = srcs1 + self.dropout2(src2)
        srcs2 = self.norm2(srcs2)

        return srcs2


class ModelBuilder(nn.Module):
    def __init__(self, label):
        super(ModelBuilder, self).__init__()

        self.backbone = backbone().cuda()  
        self.neck = AdjustAllLayer([256,256,256],[192,192,192])
        self.fusion_block3 = featureFusion(192, 6)

        self.grader = SMTN(cfg).cuda()

        self.non_local_attn = _NonLocalBlock2D(in_channels=192)
        self.cross_attn = CrossLocal(in_channels=192)

        self.adjust_attn = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )

        self.cls3loss = nn.BCEWithLogitsLoss()
        self.IOULOSS = IOULoss()

    def template(self, z, x):
        with t.no_grad():
            zf, _, _ = self.backbone.init(z)
            zf = self.neck(zf)
            zf = self.fusion(zf)
            self.zf = zf


            xf, xfeat1, xfeat2 = self.backbone.init(x)
            xf = self.neck(xf)
            xf = self.fusion(xf)
            
            non_local_attn_template = self.non_local_attn(self.zf)
            temp = xf.view(-1,xf.size(-3),xf.size(-2),xf.size(-1))
            cross_attn_template = self.cross_attn(self.zf,temp)

            non_local_attn_search = None
            cross_attn_search = None

            for i in range(x.size(0)):
                temp = xf[i,:,:,:]
                temp = t.unsqueeze(temp,0)
                temp_local = t.unsqueeze(self.non_local_attn(temp),1)
                if non_local_attn_search != None:
                    non_local_attn_search = t.cat((non_local_attn_search,temp_local),dim=1)
                else:
                    non_local_attn_search = temp_local

                temp_cross = t.unsqueeze(self.cross_attn(temp,self.zf),1)
                if cross_attn_search != None:
                    cross_attn_search = t.cat((cross_attn_search, temp_cross),dim=1)
                else:
                    cross_attn_search = temp_cross
                

            attn_template = cat((non_local_attn_template, cross_attn_template),
                                dim=1)
            attn_search = cat((non_local_attn_search, cross_attn_search), dim=2)

            attn_template = self.adjust_attn(attn_template)
            attn_search = attn_search.view(-1,attn_search.size(-3),attn_search.size(-2),attn_search.size(-1))
            attn_search = self.adjust_attn(attn_search)

            zf = attn_template.view(self.zf.size(-4), -1, self.zf.size(-2), self.zf.size(-1))

            xf = attn_search.view(
                        xf.size(-4), -1, xf.size(-2),
                        xf.size(-1))

            ppres = self.grader.conv1(self.xcorr_depthwise(xf, zf))

            self.memory = ppres
            self.featset1 = xfeat1
            self.featset2 = xfeat2

    def xcorr_depthwise(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2),
                             kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls

    def getcentercuda(self, mapp):
        def dcon(x):
            x[t.where(x <= -1)] = -0.99
            x[t.where(x >= 1)] = 0.99
            return (t.log(1 + x) - t.log(1 - x)) / 2

        size = mapp.size()[3]
        x = t.Tensor(
            np.tile((16 * (np.linspace(0, size - 1, size)) + 63) -
                    cfg.TRAIN.SEARCH_SIZE // 2, size).reshape(-1)).cuda()
        y = t.Tensor(
            np.tile(
                (16 * (np.linspace(0, size - 1, size)) + 63).reshape(-1, 1) -
                cfg.TRAIN.SEARCH_SIZE // 2, size).reshape(-1)).cuda()

        shap = dcon(mapp) * (cfg.TRAIN.SEARCH_SIZE // 2)

        xx = np.int16(
            np.tile(np.linspace(0, size - 1, size), size).reshape(-1))
        yy = np.int16(
            np.tile(np.linspace(0, size - 1, size).reshape(-1, 1),
                    size).reshape(-1))

        w = shap[:, 0, yy, xx] + shap[:, 1, yy, xx]
        h = shap[:, 2, yy, xx] + shap[:, 3, yy, xx]
        x = x - shap[:, 0, yy, xx] + w / 2 + cfg.TRAIN.SEARCH_SIZE // 2
        y = y - shap[:, 2, yy, xx] + h / 2 + cfg.TRAIN.SEARCH_SIZE // 2

        anchor = t.zeros(
            (cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.NUM_GPU, size**2, 4)).cuda()
        anchor[:, :, 0] = x - w / 2
        anchor[:, :, 1] = y - h / 2
        anchor[:, :, 2] = x + w / 2
        anchor[:, :, 3] = y + h / 2
        return anchor


    def fusion(self, feature_map_list):
            p3_temp = feature_map_list[0]
            p4_temp = feature_map_list[1]
            p5_temp = feature_map_list[2]

            b,c,w,h = p3_temp.size()
            p3 = (p3_temp).view(b,c,-1).permute(2,0,1)
            p4 = (p4_temp).view(b,c,-1).permute(2,0,1)
            p5 = (p5_temp).view(b,c,-1).permute(2,0,1)

            x1 = self.fusion_block3(p4,p3,p5).permute(1,2,0).view(b,c,w,h)
            f3 = x1
            
            return f3

    def forward(self, data):
        presearch = data['pre_search'].cuda()
        template = data['template'].cuda()
        search = data['search'].cuda()
        bbox = data['bbox'].cuda()
        labelcls2 = data['label_cls2'].cuda()
        labelxff = data['labelxff'].cuda()
        labelcls3 = data['labelcls3'].cuda()
        weightxff = data['weightxff'].cuda()

        presearch = t.cat((presearch, search.unsqueeze(1)), 1)

        zf = self.backbone(template.unsqueeze(1))
        xf = self.backbone(presearch)


        zf_temp = self.neck(zf)
        xf_temp = self.neck(xf)

        zf = self.fusion(zf_temp)
        xf = self.fusion(xf_temp)

        non_local_attn_template = self.non_local_attn(zf)
        temp = xf[:,-1,:,:,:]
        temp = temp.view(-1,xf.size(-3),xf.size(-2),xf.size(-1))
        
        cross_attn_template = self.cross_attn(zf,temp)

        non_local_attn_search = None
        cross_attn_search = None

        for i in range(cfg.TRAIN.videorange + 1):
            temp = xf[:,i,:,:,:]
            temp_local = t.unsqueeze(self.non_local_attn(temp),1)
            if non_local_attn_search != None:
                non_local_attn_search = t.cat((non_local_attn_search,temp_local),dim=1)
            else:
                non_local_attn_search = temp_local

            temp_cross = t.unsqueeze(self.cross_attn(temp,zf),1)
            if cross_attn_search != None:
                cross_attn_search = t.cat((cross_attn_search, temp_cross),dim=1)
            else:
                cross_attn_search = temp_cross
            

        attn_template = cat((non_local_attn_template, cross_attn_template),
                            dim=1)
        attn_search = cat((non_local_attn_search, cross_attn_search), dim=2)
        attn_template = self.adjust_attn(attn_template)
        attn_search = attn_search.view(-1,attn_search.size(-3),attn_search.size(-2),attn_search.size(-1))
        attn_search = self.adjust_attn(attn_search)

        zf = attn_template.view(zf.size(-4), -1, zf.size(-2), zf.size(-1))

        xf = attn_search.view(cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.NUM_GPU,
                     cfg.TRAIN.videorange + 1, -1, xf.size(-2),
                     xf.size(-1))

        loc, cls2, cls3 = self.grader(
            xf[:, -1, :, :, :], zf, xf[:, :-1, :, :, :].permute(1, 0, 2, 3, 4))

        cls2 = self.log_softmax(cls2)


        cls_loss2 = select_cross_entropy_loss(cls2, labelcls2)
        cls_loss3 = self.cls3loss(cls3, labelcls3)

        pre_bbox = self.getcentercuda(loc)
        bbo = self.getcentercuda(labelxff)

        loc_loss1 = self.IOULOSS(pre_bbox, bbo, weightxff)
        loc_loss2 = DISCLE(pre_bbox, bbo, weightxff)
        loc_loss = cfg.TRAIN.w2 * loc_loss1 + cfg.TRAIN.w3 * loc_loss2
        cls_loss = cfg.TRAIN.w4 * cls_loss2 + cfg.TRAIN.w5 * cls_loss3

        outputs = {}
        outputs['total_loss'] =\
            cfg.TRAIN.LOC_WEIGHT*loc_loss\
                +cfg.TRAIN.CLS_WEIGHT*cls_loss

        outputs['cls_loss'] = cls_loss
        outputs['loc_loss1'] = loc_loss1
        outputs['loc_loss2'] = loc_loss2

        return outputs
