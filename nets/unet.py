import torch
import torch.nn as nn
from nets.resnet import resnet50
from nets.vgg import VGG16
from torchinfo import summary
from nets.swin_transformer import New_model
#特征融合模块
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)             #torch.cat将两个张量（tensor）拼接在一起
                                                                        #C=torch.cat((A,B),1)就表示按维数1（列）拼接A和B，也就是横着拼接，A左B右
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 2, pretrained = False, backbone = 'resnet50', embed_dim = 96):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]                            #[64+128,128+256,256+512,]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            # in_filters = [192, 512, 1024, 3072]                          #不加swin_tran时使用
            in_filters = [224, 512, 1024, 3072]                            #只有一个swin块
            # in_filters  = [224,448,896,2816]                      #4层swin块[96+128,192+256,384+512,768+2048]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        # swin_in_size = [256, 128, 64, 32]
        swin_in_size = [(112,224), (56,112), (28,56), (14,28)]
        swin_in_filters = [64, 256, 512, 1024]
        num_heads = [3,6,12,24]
        num_blocks = [2, 2, 6, 2]


        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])


        self.swin_transformer1 = New_model(img_size=swin_in_size[0], in_chans=swin_in_filters[0],                #由于New_model中depths和num_heads必须是列表格式，所以需要num_blocks[i]，num_heads[0]多加[]形成列表
                                           embed_dim=embed_dim, depths=[num_blocks[0]], num_heads=[num_heads[0]])
        self.swin_transformer2 = New_model(img_size=swin_in_size[1], in_chans=swin_in_filters[1],
                                           embed_dim=embed_dim*2, depths=[num_blocks[1]], num_heads=[num_heads[1]])
        self.swin_transformer3 = New_model(img_size=swin_in_size[2], in_chans=swin_in_filters[2],
                                           embed_dim=embed_dim*4, depths=[num_blocks[2]], num_heads=[num_heads[2]])
        self.swin_transformer4 = New_model(img_size=swin_in_size[3], in_chans=swin_in_filters[3],
                                           embed_dim=embed_dim*8, depths=[num_blocks[3]], num_heads=[num_heads[3]])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)                 # feat1 = (1,64,256,256)
        elif self.backbone == "resnet50":                                                  # feat2 = (1,256,128,128)
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)              # feat3 = (1,512,64,64)
                                                                                           # feat4 = (1,1024,32,32)
                                                                                                       # feat5 = (1,2048,16,16)
        swintran_out1 = self.swin_transformer1(feat1)
        # swintran_out2 = self.swin_transformer2(feat2)
        # swintran_out3 = self.swin_transformer3(feat3)
        # swintran_out4 = self.swin_transformer4(feat4)

        # up4 = self.up_concat4(swintran_out4, feat5)                                                # up4 = (1, 512, 32, 32)
        # up3 = self.up_concat3(swintran_out3, up4)                                                  # up3 = (1, 256, 64, 64)
        # up2 = self.up_concat2(swintran_out2, up3)                                                  # up2 = (1, 128, 128, 128)
        # up1 = self.up_concat1(swintran_out1, up2)                                                  # up1 = (1, 64, 256, 256)

        up4 = self.up_concat4(feat4, feat5)                                                # up4 = (1, 512, 32, 32)
        up3 = self.up_concat3(feat3, up4)                                                  # up3 = (1, 256, 64, 64)
        up2 = self.up_concat2(feat2, up3)                                                  # up2 = (1, 128, 128, 128)
        up1 = self.up_concat1(swintran_out1, up2)
        # up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:                   #为什么backbone == 'resnet50'要再进行一次上采样？？？  答： 恢复原来的512的大小
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


# net = Unet().cuda()
# batch_size = 1
# summary(net, input_size=(batch_size, 3, 224, 448))
# print(summary)
