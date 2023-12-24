import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class Swinv2_backbone(nn.Module):
    def __init__(self):
        super(Swinv2_backbone, self).__init__()

        # 创建 Swin Transformer 模型，并加载预训练权重
        self.swin = timm.create_model('swinv2_base_window8_256.ms_in1k', pretrained=False)

        # 提取 Swin Transformer 的前三层
        self.pe = self.swin.patch_embed
        self.layer1 = self.swin.layers[0]
        self.layer2 = self.swin.layers[1]
        self.layer3 = self.swin.layers[2]
        self.layer4 = self.swin.layers[3]

    def forward(self, x):
        # 特定层的前向传播
        outs = []
        x = self.pe(x)
        x = self.layer1(x)
        # B, H, W, C = x.shape
        x1 = x.permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        ###########################################

        x = self.layer2(x)
        # B, H, W, C = x.shape
        x2 = x.permute(0, 3, 1, 2).contiguous()
        outs.append(x2)
        ###################################

        x = self.layer3(x)
        # B, H, W, C = x.shape
        x3 = x.permute(0, 3, 1, 2).contiguous()
        outs.append(x3)
        #############################################
        x = self.layer4(x)
        # B, H, W, C = x.shape
        x4 = x.permute(0, 3, 1, 2).contiguous()
        outs.append(x4)

        return outs




class deeplabSWINFormer(nn.Module):
    def __init__(self, num_classes=21, pretrained=False):
        super(deeplabSWINFormer, self).__init__()
        self.in_channels = [128, 256, 512, 1024]
##############################主干################################
        self.backbone = Swinv2_backbone()
#######################分割头###################################
        self.aspp = ASPP(dim_in=1024, dim_out=256, rate=16 // 8)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(128, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        low_level_features, high_feature = x[0], x[3]
        high_feature = self.aspp(high_feature)

        low_level_features = self.shortcut_conv(low_level_features)

        high_feature = F.interpolate(high_feature, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((high_feature, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    model = deeplabSWINFormer(num_classes=2)
    # 定义随机输入数据
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    input_tensor = torch.randn(batch_size, channels, height, width)
    #
    # 运行模型推理
    with torch.no_grad():
        output = model(input_tensor)
    #
    # # 打印输出张量的形状
    print(output)
