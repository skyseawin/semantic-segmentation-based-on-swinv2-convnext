import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class Swinv2_backbone(nn.Module):
    def __init__(self):
        super(Swinv2_backbone, self).__init__()

        # 创建 Swin Transformer 模型，并加载预训练权重
        self.swin = timm.create_model('swinv2_base_window8_256.ms_in1k', pretrained=False)

        # 提取 Swin Transformer 
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


class SegSWINFormer(nn.Module):
    def __init__(self, num_classes=21, pretrained=False):
        super(SegSWINFormer, self).__init__()
        self.in_channels = [128, 256, 512, 1024]
##########################主干##############################################
        self.backbone = Swinv2_backbone()
##########################分割头###############################################
        self.embedding_dim = 768
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x




if __name__ == '__main__':
    model = SegSWINFormer(num_classes=2)
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
