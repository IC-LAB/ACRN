import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

# FPT 
from FPT_tansformer.self_trans import SelfTrans
from FPT_tansformer.rendering_trans import RenderTrans
from FPT_tansformer.grounding_trans import GroundTrans 
# from dropblock import DropBlock2D


# from torch.utils.model_zoo import load_url as load_state_dict_from_url
#  resnet / vgg/ inception refer to : https://github.com/pytorch/vision/tree/master/torchvision/models

model_urls = { 
    'res18_cls_res': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 
    'res50_cls_res': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 
    'res18_cls': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 
    'res50_cls': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 
    'vgg16_cls': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_cls': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'vgg16_cls_res': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_cls_res': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'FPN_cls': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 
    'FPN_seg': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class res50_cls_res(nn.Module):

    def __init__(self, transform_input=False, cls_res=False, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(res50_cls_res, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.cls_res = cls_res
        self.transform_input = transform_input
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc_new = nn.Linear(512 * block.expansion, num_classes)
        self.fc_regress = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_class = self.fc_new(x)
        
        # return x
        # add softmax
        # x = nn.Softmax(x)
        if self.cls_res:
            x_regress = self.fc_regress(x)
            return x_class,x_regress
        else:
            return x_class

class FPN_seg(nn.Module):

    def __init__(self, transform_input=False,block=Bottleneck, layers=[3, 4, 6, 3], num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(FPN_seg, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.transform_input = transform_input
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        # self.fc_new = nn.Linear(512 * block.expansion, num_classes)
        # self.fc_regress = nn.Linear(512 * block.expansion, 1)

        ###FPN
        
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128*4, 2, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # fc = self.avgpool(c5)
        # fc_ = torch.flatten(fc, 1)
        # x_class = self.fc_new(fc_)
        # x_regress = self.fc_regress(fc_)
        # return x
        # add softmax
        # x = nn.Softmax(x)
        
        #scal_fpn 
        '''https://github.com/ElephantGit/SemanticSegmentationUsingFPN_PanopticFeaturePyramidNetworks/blob/master/model/FPN.py
        '''
        
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h//4, w//4)
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h//2, w//2)
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h//2, w//2)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        seg_mask = self._upsample(self.conv3(torch.cat([s2 ,s3 ,s4 ,s5],1)), 4 * h, 4 * w)
         
        return seg_mask #x_class,x_regress

class FPN_cls_seg(nn.Module):

    def __init__(self, transform_input=False, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(FPN_cls_seg, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.transform_input = transform_input
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print("block.expansion",block.expansion) =4
        self.fc_new = nn.Linear(512 * block.expansion, num_classes)
        self.fc_regress = nn.Linear(512 * block.expansion, 1)

        ###FPN
        
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128*4, 2, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        fc = self.avgpool(c5)
        fc_ = torch.flatten(fc, 1)
        x_class = self.fc_new(fc_)
        # return x_class
        # x_regress = self.fc_regress(fc_)
        # return x
        # add softmax
        # x = nn.Softmax(x)
        #scal_fpn 
        '''https://github.com/ElephantGit/SemanticSegmentationUsingFPN_PanopticFeaturePyramidNetworks/blob/master/model/FPN.py
        '''
        
        # Top-down
        p5 = self.toplayer(c5) #256
        p4 = self._upsample_add(p5, self.latlayer1(c4))#256
        p3 = self._upsample_add(p4, self.latlayer2(c3))#256
        p2 = self._upsample_add(p3, self.latlayer3(c2))#256
         
        # return p2,p3,p4,p5
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h//4, w//4)
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h//2, w//2)
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h//2, w//2)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        seg_mask = self._upsample(self.conv3(torch.cat([s2 ,s3 ,s4 ,s5],1)), 4 * h, 4 * w)
        
        return x_class,seg_mask #x_regress


class FPT_cls_seg(nn.Module):

    def __init__(self, transform_input=False, cls_seg=False, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,
                 drop=False, upsample_method = 'bilinear', with_norm = 'batch_norm',feature_dim = 256):
        super(FPT_cls_seg, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.drop = drop
        self.cls_seg = cls_seg

        self.inplanes = 64
        self.dilation = 1
        self.transform_input = transform_input
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print("block.expansion",block.expansion) =4
        if self.cls_seg:
            self.fc_new = nn.Linear(512 * block.expansion, num_classes)
            self.fc_regress = nn.Linear(512 * block.expansion, 1)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128*4, 2, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)

        # FPT 
        self.feature_dim = feature_dim #256 or 128

        # upsample_method = upsample_method # 'bilinear'
        assert upsample_method in ['nearest', 'bilinear']
        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method, align_corners=False if upsample_method == 'bilinear' else None)
        self.fpn_upsample = interpolate
        
        # with_norm = with_norm #'none'  
        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

        self.st_p5 = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim)
        self.st_p4 = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim)
        self.st_p3 = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim)
        self.st_p2 = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim)
        
        self.gt_p4_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p3_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p3_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p3 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        
        self.rt_p5_p4 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p3 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p3 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p3_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        if self.drop:
            self.drop_block = DropBlock2D(block_size=3, drop_prob=0.2)
        
        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])
            
            self.fpt_p5 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p4 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p3 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p2 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
        else:
            self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)
            
            self.fpt_p5 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p4 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p3 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p2 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        if self.cls_seg:
            fc = self.avgpool(c5)
            fc_ = torch.flatten(fc, 1)
            x_class = self.fc_new(fc_)
        # x_regress = self.fc_regress(fc_)
        # return x
        # add softmax
        # x = nn.Softmax(x)

        # FPT net  
        fpn_p5_1 = self.fpn_p5_1x1(c5) #8
        fpn_p4_1 = self.fpn_p4_1x1(c4) #16
        fpn_p3_1 = self.fpn_p3_1x1(c3) #32
        fpn_p2_1 = self.fpn_p2_1x1(c2) #64 

        fpt_p5_out = torch.cat((self.st_p5(fpn_p5_1), 
                                self.rt_p5_p4(fpn_p5_1, fpn_p4_1), self.rt_p5_p3(fpn_p5_1,fpn_p3_1), self.rt_p5_p2(fpn_p5_1,fpn_p2_1), 
                                fpn_p5_1), 1)
        fpt_p4_out = torch.cat((self.st_p4(fpn_p4_1), 
                                self.rt_p4_p3(fpn_p4_1, fpn_p3_1), self.rt_p4_p2(fpn_p4_1,fpn_p2_1), 
                                self.gt_p4_p5(fpn_p4_1,fpn_p5_1), 
                                fpn_p4_1), 1)
        fpt_p3_out = torch.cat((self.st_p3(fpn_p3_1), 
                                self.rt_p3_p2(fpn_p3_1, fpn_p2_1), 
                                self.gt_p3_p4(fpn_p3_1,fpn_p4_1), self.gt_p3_p5(fpn_p3_1,fpn_p5_1), 
                                fpn_p3_1), 1)
        fpt_p2_out = torch.cat((self.st_p2(fpn_p2_1), 
                                self.gt_p2_p3(fpn_p2_1, fpn_p3_1), self.gt_p2_p4(fpn_p2_1,fpn_p4_1), self.gt_p2_p5(fpn_p2_1,fpn_p5_1), 
                                fpn_p2_1), 1)

        p5 = self.fpt_p5(fpt_p5_out) # N*128*8*8
        p4 = self.fpt_p4(fpt_p4_out) # N*128*16*16
        p3 = self.fpt_p3(fpt_p3_out) # N*128*32*32
        p2 = self.fpt_p2(fpt_p2_out) # N*128*64*64
        if self.drop:
            p5 = self.drop_block(p5)
            p4 = self.drop_block(p4)
            p3 = self.drop_block(p3)
            p2 = self.drop_block(p2)  

        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h//4, w//4)
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h//2, w//2)
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h//2, w//2)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        seg_mask = self._upsample(self.conv3(torch.cat([s2 ,s3 ,s4 ,s5],1)), 4 * h, 4 * w)
         
        if self.cls_seg:
            return x_class,seg_mask #x_regress 
        else:
            return seg_mask


# class FPN_cls_cycle(nn.Module):

    # def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None):
    #     super(FPN_cls_cycle, self).__init__()
    #     if norm_layer is None:
    #         norm_layer = nn.BatchNorm2d
    #     self._norm_layer = norm_layer

    #     self.inplanes = 64
    #     self.dilation = 1
    #     if replace_stride_with_dilation is None:
    #         # each element in the tuple indicates if we should replace
    #         # the 2x2 stride with a dilated convolution instead
    #         replace_stride_with_dilation = [False, False, False]
    #     if len(replace_stride_with_dilation) != 3:
    #         raise ValueError("replace_stride_with_dilation should be None "
    #                          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    #     self.groups = groups
    #     self.base_width = width_per_group
    #     self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    #     self.bn1 = norm_layer(self.inplanes)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #     self.layer1 = self._make_layer(block, 64, layers[0])
    #     self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
    #                                    dilate=replace_stride_with_dilation[0])
    #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
    #                                    dilate=replace_stride_with_dilation[1])
    #     self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
    #                                    dilate=replace_stride_with_dilation[2])
    #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
    #     self.fc_class = nn.Linear(512 * block.expansion, num_classes)
    #     self.fc_regress = nn.Linear(512 * block.expansion, 1)
        
    #     self.fc_class_seg = nn.Linear(num_classes*2, num_classes)

    #     ###FPN
        
    #     # Top layer
    #     self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

    #     # Smooth layers
    #     self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    #     # Lateral layers
    #     self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

	# 	# Semantic branch
    #     self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(128*4, num_classes, kernel_size=1, stride=1, padding=0)
    #     self.conv4_2 = nn.Conv2d(num_classes*2, num_classes, kernel_size=1, stride=1, padding=0)

    #     # num_groups, num_channels
    #     self.gn1 = nn.GroupNorm(128, 128) 
    #     self.gn2 = nn.GroupNorm(256, 256)


    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    #     # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    #     if zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.bn3.weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.bn2.weight, 0)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)
    # def _upsample(self, x, h, w):
    #     return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    # def _upsample_add(self, x, y):
    #     '''Upsample and add two feature maps.
    #     Args:
    #       x: (Variable) top feature map to be upsampled.
    #       y: (Variable) lateral feature map.
    #     Returns:
    #       (Variable) added feature map.
    #     Note in PyTorch, when input size is odd, the upsampled feature map
    #     with `F.upsample(..., scale_factor=2, mode='nearest')`
    #     maybe not equal to the lateral feature map size.
    #     e.g.
    #     original input size: [N,_,15,15] ->
    #     conv2d feature map size: [N,_,8,8] ->
    #     upsampled feature map size: [N,_,16,16]
    #     So we choose bilinear upsample which supports arbitrary output sizes.
    #     '''
    #     _,_,H,W = y.size()
    #     return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     c1 = self.maxpool(x)

    #     c2 = self.layer1(c1)
    #     c3 = self.layer2(c2)
    #     c4 = self.layer3(c3)
    #     c5 = self.layer4(c4)
    #     fc = self.avgpool(c5)
    #     fc_ = torch.flatten(fc, 1)
    #     x_class_ = self.fc_class(fc_)
    #     # x_regress = self.fc_regress(fc_)
    #     # return x
    #     # add softmax
    #     # x = nn.Softmax(x)
    #     #scal_fpn 
    #     '''https://github.com/ElephantGit/SemanticSegmentationUsingFPN_PanopticFeaturePyramidNetworks/blob/master/model/FPN.py
    #     '''
        
    #     # Top-down
    #     p5 = self.toplayer(c5)
    #     p4 = self._upsample_add(p5, self.latlayer1(c4))
    #     p3 = self._upsample_add(p4, self.latlayer2(c3))
    #     p2 = self._upsample_add(p3, self.latlayer3(c2))

    #     # Smooth
    #     p4 = self.smooth1(p4)
    #     p3 = self.smooth2(p3)
    #     p2 = self.smooth3(p2)

    #     # Semantic
    #     _, _, h, w = p2.size()
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h//4, w//4)
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h//2, w//2)
    #     # 256->128
    #     s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

    #     # 256->256
    #     s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h//2, w//2)
    #     # 256->128
    #     s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

    #     # 256->128
    #     s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

    #     s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        
    #     seg_2_mask = self.conv3(torch.cat([s2 ,s3 ,s4 ,s5],1))
        
    #     #cycle
    #     # seg concat to class
    #     seg_class = torch.flatten(self.avgpool(seg_2_mask), 1) #seg计算概率
    #     x_class = self.fc_class_seg(torch.cat([x_class_, seg_class],1))

    #     #  class concat to seg
    #     n,c = x_class.size()
    #     _,_,h,w = seg_2_mask.size()
    #     class_2_mask = (torch.unsqueeze(torch.unsqueeze(x_class_,2),3)).expand(n,c,h,w)       
    #     seg_mask = self._upsample(self.conv4_2(torch.cat([seg_2_mask,class_2_mask],1)), 4 * h, 4 * w)
        
    #     return x_class,seg_mask #x_regress
    #     # return c2,c3,c4,c5,p2,p3,p4,p5,s2,s3,s4,s5,\
    #     #     seg_2_mask,seg_class,x_class,class_2_mask,seg_mask
        
    #     # # seg_2_mask = self.conv3(torch.cat([s2 ,s3 ,s4 ,s5],1))
    #     # # seg_2_mask_softmax = F.softmax(seg_2_mask,dim=1)
        
    #     # # #cycle
    #     # # # seg concat to class
    #     # # seg_class = torch.flatten(self.avgpool(seg_2_mask_softmax), 1) #seg计算概率
    #     # # x_class_softmax = F.softmax(x_class_,dim=1)
    #     # # x_class = self.fc_class_seg(torch.cat([x_class_softmax, seg_class],1))

    #     # # #  class concat to seg
    #     # # n,c = x_class_softmax.size()
    #     # # _,_,h,w = seg_2_mask_softmax.size()
    #     # # class_2_mask_softmax = (torch.unsqueeze(torch.unsqueeze(x_class_softmax,2),3)).expand(n,c,h,w)       
    #     # # seg_mask = self._upsample(self.conv4_2(torch.cat([seg_2_mask_softmax,class_2_mask_softmax],1)), 4 * h, 4 * w)
        
    #     # # return x_class,seg_mask #x_regress
    #     # # # return c2,c3,c4,c5,p2,p3,p4,p5,s2,s3,s4,s5,\
    #     # # #     seg_2_mask,seg_class,x_class,class_2_mask_softmax,seg_mask

# class FPN_cls_gap_cycle_128(nn.Module):
    # '''gap, 128, cycle'''
    # def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None):
    #     super(FPN_cls_gap_cycle_128, self).__init__()
    #     if norm_layer is None:
    #         norm_layer = nn.BatchNorm2d
    #     self._norm_layer = norm_layer

    #     self.inplanes = 64
    #     self.dilation = 1
    #     if replace_stride_with_dilation is None:
    #         # each element in the tuple indicates if we should replace
    #         # the 2x2 stride with a dilated convolution instead
    #         replace_stride_with_dilation = [False, False, False]
    #     if len(replace_stride_with_dilation) != 3:
    #         raise ValueError("replace_stride_with_dilation should be None "
    #                          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    #     self.groups = groups
    #     self.base_width = width_per_group
    #     self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    #     self.bn1 = norm_layer(self.inplanes)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #     self.layer1 = self._make_layer(block, 64, layers[0])
    #     self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
    #                                    dilate=replace_stride_with_dilation[0])
    #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
    #                                    dilate=replace_stride_with_dilation[1])
    #     self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
    #                                    dilate=replace_stride_with_dilation[2])
    #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
    #     self.fc_class_128 = nn.Linear(512 * block.expansion, 128)
    #     self.fc_class_2 = nn.Linear(256,2)
    #     # self.fc_regress = nn.Linear(512 * block.expansion, 1)
        
    #     self.fc_class_seg = nn.Linear(256, num_classes)

    #     ###FPN
        
    #     # Top layer
    #     self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

    #     # Smooth layers
    #     self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    #     # Lateral layers
    #     self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

	# 	# Semantic branch
    #     self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(128*4, 1, kernel_size=1, stride=1, padding=0)
    #     self.conv128 = nn.Conv2d(128*4, 128, kernel_size=1, stride=1, padding=0)
    #     self.conv256 = nn.Conv2d(128*2, num_classes, kernel_size=1, stride=1, padding=0) 
    #     # num_groups, num_channels
    #     self.gn1 = nn.GroupNorm(128, 128) 
    #     self.gn2 = nn.GroupNorm(256, 256)

    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    #     # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    #     if zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.bn3.weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.bn2.weight, 0)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)
    # def _upsample(self, x, h, w):
    #     return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    # def _upsample_add(self, x, y):
    #     '''Upsample and add two feature maps.
    #     Args:
    #       x: (Variable) top feature map to be upsampled.
    #       y: (Variable) lateral feature map.
    #     Returns:
    #       (Variable) added feature map.
    #     Note in PyTorch, when input size is odd, the upsampled feature map
    #     with `F.upsample(..., scale_factor=2, mode='nearest')`
    #     maybe not equal to the lateral feature map size.
    #     e.g.
    #     original input size: [N,_,15,15] ->
    #     conv2d feature map size: [N,_,8,8] ->
    #     upsampled feature map size: [N,_,16,16]
    #     So we choose bilinear upsample which supports arbitrary output sizes.
    #     '''
    #     _,_,H,W = y.size()
    #     return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     c1 = self.maxpool(x)

    #     c2 = self.layer1(c1)
    #     c3 = self.layer2(c2)
    #     c4 = self.layer3(c3)
    #     c5 = self.layer4(c4)
    #     fc = self.avgpool(c5)
    #     fc = torch.flatten(fc, 1)
    #     x_class_128 = self.fc_class_128(fc)
    #     # x_regress = self.fc_regress(fc)

        
    #     # return x
    #     # add softmax
    #     # x = nn.Softmax(x)
    #     #scal_fpn 
    #     '''https://github.com/ElephantGit/SemanticSegmentationUsingFPN_PanopticFeaturePyramidNetworks/blob/master/model/FPN.py
    #     '''
        
    #     # Top-down
    #     p5 = self.toplayer(c5)
    #     p4 = self._upsample_add(p5, self.latlayer1(c4))
    #     p3 = self._upsample_add(p4, self.latlayer2(c3))
    #     p2 = self._upsample_add(p3, self.latlayer3(c2))

    #     # Smooth
    #     p4 = self.smooth1(p4)
    #     p3 = self.smooth2(p3)
    #     p2 = self.smooth3(p2)

    #     # Semantic
    #     _, _, h, w = p2.size()
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h//4, w//4)
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h//2, w//2)
    #     # 256->128
    #     s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

    #     # 256->256
    #     s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h//2, w//2)
    #     # 256->128
    #     s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

    #     # 256->128
    #     s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

    #     s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        
    #     seg_128_mask = self.conv128(torch.cat([s2 ,s3 ,s4 ,s5],1))
        
    #     # seg concat to class
    #     seg_class = torch.flatten(self.avgpool(seg_128_mask), 1) #seg计算概率
    #     x_class_final = self.fc_class_seg(torch.cat([x_class_128, seg_class],1))

    #     #  class concat to seg
    #     n,c = x_class_128.size()
    #     _,_,h,w = seg_128_mask.size()
    #     class_128_mask = (torch.unsqueeze(torch.unsqueeze(x_class_128,2),3)).expand(n,c,h,w) 
          
    #     seg_mask = self._upsample(self.conv256(torch.cat([class_128_mask, seg_128_mask],1)), 4 * h, 4 * w)
        
    #     # return x_class_final,x_regress,seg_mask
    #     # return c2,c3,c4,c5,p2,p3,p4,p5,s2,s3,s4,s5,seg_128_mask,seg_class,x_class_128,x_class_final,class_128_mask,seg_mask
    #     return x_class_final,seg_mask #x_regress

# class FPN_cls_flatten_cycle_256(nn.Module):
    # '''fpn,flatten, 256'''
    # def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None):
    #     super(FPN_cls_flatten_cycle_256, self).__init__()
    #     if norm_layer is None:
    #         norm_layer = nn.BatchNorm2d
    #     self._norm_layer = norm_layer

    #     self.inplanes = 64
    #     self.dilation = 1
    #     if replace_stride_with_dilation is None:
    #         # each element in the tuple indicates if we should replace
    #         # the 2x2 stride with a dilated convolution instead
    #         replace_stride_with_dilation = [False, False, False]
    #     if len(replace_stride_with_dilation) != 3:
    #         raise ValueError("replace_stride_with_dilation should be None "
    #                          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    #     self.groups = groups
    #     self.base_width = width_per_group
    #     self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    #     self.bn1 = norm_layer(self.inplanes)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #     self.layer1 = self._make_layer(block, 64, layers[0])
    #     self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
    #                                    dilate=replace_stride_with_dilation[0])
    #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
    #                                    dilate=replace_stride_with_dilation[1])
    #     self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
    #                                    dilate=replace_stride_with_dilation[2])
    #     self.avgpool = nn.AdaptiveAvgPool2d((16,16)) 
    #     # self.fc_class = nn.Linear(512 * block.expansion, num_classes)
    #     # self.fc_regress = nn.Linear(512 * block.expansion, 1)
    #     self.fc_class_seg = nn.Linear(256+256, num_classes)
 
    #     self.conv2048=conv1x1(2048,1)

    #     ###FPN
        
    #     # Top layer
    #     self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

    #     # Smooth layers
    #     self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    #     # Lateral layers
    #     self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

	# 	# Semantic branch
    #     self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(128*4, 1, kernel_size=1, stride=1, padding=0)
    #     self.conv2_1 = nn.Conv2d(2, num_classes, kernel_size=1, stride=1, padding=0)
    #     # num_groups, num_channels
    #     self.gn1 = nn.GroupNorm(128, 128) 
    #     self.gn2 = nn.GroupNorm(256, 256)


    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    #     # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    #     if zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.bn3.weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.bn2.weight, 0)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)
    # def _upsample(self, x, h, w):
    #     return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    # def _upsample_add(self, x, y):
    #     '''Upsample and add two feature maps.
    #     Args:
    #       x: (Variable) top feature map to be upsampled.
    #       y: (Variable) lateral feature map.
    #     Returns:
    #       (Variable) added feature map.
    #     Note in PyTorch, when input size is odd, the upsampled feature map
    #     with `F.upsample(..., scale_factor=2, mode='nearest')`
    #     maybe not equal to the lateral feature map size.
    #     e.g.
    #     original input size: [N,_,15,15] ->
    #     conv2d feature map size: [N,_,8,8] ->
    #     upsampled feature map size: [N,_,16,16]
    #     So we choose bilinear upsample which supports arbitrary output sizes.
    #     '''
    #     _,_,H,W = y.size()
    #     return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     c1 = self.maxpool(x)

    #     c2 = self.layer1(c1)
    #     c3 = self.layer2(c2)
    #     c4 = self.layer3(c3)
    #     c5 = self.layer4(c4)
    #     # fc = self.avgpool(c5)
    #     # fc_ = torch.flatten(fc, 1)
    #     # x_class = self.fc_class(fc_)
    #     # x_regress = self.fc_regress(fc_)
    #     c6=self.conv2048(c5)
    #     fc = torch.flatten(c6, 1)
        
    #     # return x
    #     # add softmax
    #     # x = nn.Softmax(x)
    #     #scal_fpn 
    #     '''https://github.com/ElephantGit/SemanticSegmentationUsingFPN_PanopticFeaturePyramidNetworks/blob/master/model/FPN.py
    #     '''
        
    #     # Top-down
    #     p5 = self.toplayer(c5)
    #     p4 = self._upsample_add(p5, self.latlayer1(c4))
    #     p3 = self._upsample_add(p4, self.latlayer2(c3))
    #     p2 = self._upsample_add(p3, self.latlayer3(c2))

    #     # Smooth
    #     p4 = self.smooth1(p4)
    #     p3 = self.smooth2(p3)
    #     p2 = self.smooth3(p2)

    #     # Semantic
    #     _, _, h, w = p2.size()
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h//4, w//4)
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h//2, w//2)
    #     # 256->128
    #     s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

    #     # 256->256
    #     s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h//2, w//2)
    #     # 256->128
    #     s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

    #     # 256->128
    #     s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

    #     s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        
    #     seg_1_mask = self.conv3(torch.cat([s2 ,s3 ,s4 ,s5],1))
         

    #     # seg concat to class
    #     seg_class = torch.flatten(self.avgpool(seg_1_mask), 1) #seg计算概率
    #     x_class = self.fc_class_seg(torch.cat([fc, seg_class],1))

    #     #  class concat to seg 
    #     _,_,h,w = seg_1_mask.size() 
    #     class_1_mask =  self._upsample(c6 , h, w)
      
    #     seg_mask = self._upsample(self.conv2_1(torch.cat([class_1_mask, seg_1_mask],1)), 4 * h, 4 * w)
        
    #     return x_class,seg_mask

# class FPN_cls_flatten_cycle_2048(nn.Module):
    # '''fpn,flatten, 2048'''
    # def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None):
    #     super(FPN_cls_flatten_cycle_2048, self).__init__()
    #     if norm_layer is None:
    #         norm_layer = nn.BatchNorm2d
    #     self._norm_layer = norm_layer

    #     self.inplanes = 64
    #     self.dilation = 1
    #     if replace_stride_with_dilation is None:
    #         # each element in the tuple indicates if we should replace
    #         # the 2x2 stride with a dilated convolution instead
    #         replace_stride_with_dilation = [False, False, False]
    #     if len(replace_stride_with_dilation) != 3:
    #         raise ValueError("replace_stride_with_dilation should be None "
    #                          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    #     self.groups = groups
    #     self.base_width = width_per_group
    #     self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    #     self.bn1 = norm_layer(self.inplanes)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #     self.layer1 = self._make_layer(block, 64, layers[0])
    #     self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
    #                                    dilate=replace_stride_with_dilation[0])
    #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
    #                                    dilate=replace_stride_with_dilation[1])
    #     self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
    #                                    dilate=replace_stride_with_dilation[2])
    #     self.avgpool = nn.AdaptiveAvgPool2d((16,16)) 
    #     # self.fc_class = nn.Linear(512 * block.expansion, num_classes)
    #     # self.fc_regress = nn.Linear(512 * block.expansion, 1)
    #     self.fc_class_seg = nn.Linear(2048+2048, num_classes)
 
    #     self.conv2048=conv1x1(2048,8)

    #     ###FPN
        
    #     # Top layer
    #     self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

    #     # Smooth layers
    #     self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    #     # Lateral layers
    #     self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

	# 	# Semantic branch
    #     self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(128*4, 8, kernel_size=1, stride=1, padding=0)
    #     self.conv2_1 = nn.Conv2d(8+8, num_classes, kernel_size=1, stride=1, padding=0)
    #     # num_groups, num_channels
    #     self.gn1 = nn.GroupNorm(128, 128) 
    #     self.gn2 = nn.GroupNorm(256, 256)


    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    #     # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    #     if zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.bn3.weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.bn2.weight, 0)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)
    # def _upsample(self, x, h, w):
    #     return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    # def _upsample_add(self, x, y):
    #     '''Upsample and add two feature maps.
    #     Args:
    #       x: (Variable) top feature map to be upsampled.
    #       y: (Variable) lateral feature map.
    #     Returns:
    #       (Variable) added feature map.
    #     Note in PyTorch, when input size is odd, the upsampled feature map
    #     with `F.upsample(..., scale_factor=2, mode='nearest')`
    #     maybe not equal to the lateral feature map size.
    #     e.g.
    #     original input size: [N,_,15,15] ->
    #     conv2d feature map size: [N,_,8,8] ->
    #     upsampled feature map size: [N,_,16,16]
    #     So we choose bilinear upsample which supports arbitrary output sizes.
    #     '''
    #     _,_,H,W = y.size()
    #     return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     c1 = self.maxpool(x)

    #     c2 = self.layer1(c1)
    #     c3 = self.layer2(c2)
    #     c4 = self.layer3(c3)
    #     c5 = self.layer4(c4)
    #     # fc = self.avgpool(c5)
    #     # fc_ = torch.flatten(fc, 1)
    #     # x_class = self.fc_class(fc_)
    #     # x_regress = self.fc_regress(fc_)
    #     c6=self.conv2048(c5)
    #     fc = torch.flatten(c6, 1)
        
    #     # return x
    #     # add softmax
    #     # x = nn.Softmax(x)
    #     #scal_fpn 
    #     '''https://github.com/ElephantGit/SemanticSegmentationUsingFPN_PanopticFeaturePyramidNetworks/blob/master/model/FPN.py
    #     '''
        
    #     # Top-down
    #     p5 = self.toplayer(c5)
    #     p4 = self._upsample_add(p5, self.latlayer1(c4))
    #     p3 = self._upsample_add(p4, self.latlayer2(c3))
    #     p2 = self._upsample_add(p3, self.latlayer3(c2))

    #     # Smooth
    #     p4 = self.smooth1(p4)
    #     p3 = self.smooth2(p3)
    #     p2 = self.smooth3(p2)

    #     # Semantic
    #     _, _, h, w = p2.size()
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h//4, w//4)
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h//2, w//2)
    #     # 256->128
    #     s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

    #     # 256->256
    #     s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h//2, w//2)
    #     # 256->128
    #     s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

    #     # 256->128
    #     s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

    #     s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        
    #     seg_1_mask = self.conv3(torch.cat([s2 ,s3 ,s4 ,s5],1))
         

    #     # seg concat to class
    #     seg_class = torch.flatten(self.avgpool(seg_1_mask), 1) #seg计算概率
    #     x_class = self.fc_class_seg(torch.cat([fc, seg_class],1))

    #     #  class concat to seg 
    #     _,_,h,w = seg_1_mask.size() 
    #     class_1_mask =  self._upsample(c6 , h, w)
      
    #     seg_mask = self._upsample(self.conv2_1(torch.cat([class_1_mask, seg_1_mask],1)), 4 * h, 4 * w)
        
    #     return x_class,seg_mask

# class FPN_cls_s5(nn.Module):
    # '''简单地进行分类，分割，直接将c5输出上采样分割'''

    # def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None):
    #     super(FPN_cls_s5, self).__init__()
    #     if norm_layer is None:
    #         norm_layer = nn.BatchNorm2d
    #     self._norm_layer = norm_layer

    #     self.inplanes = 64
    #     self.dilation = 1
    #     if replace_stride_with_dilation is None:
    #         # each element in the tuple indicates if we should replace
    #         # the 2x2 stride with a dilated convolution instead
    #         replace_stride_with_dilation = [False, False, False]
    #     if len(replace_stride_with_dilation) != 3:
    #         raise ValueError("replace_stride_with_dilation should be None "
    #                          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    #     self.groups = groups
    #     self.base_width = width_per_group
    #     self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    #     self.bn1 = norm_layer(self.inplanes)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #     self.layer1 = self._make_layer(block, 64, layers[0])
    #     self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
    #                                    dilate=replace_stride_with_dilation[0])
    #     self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
    #                                    dilate=replace_stride_with_dilation[1])
    #     self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
    #                                    dilate=replace_stride_with_dilation[2])
    #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    #     # print("block.expansion",block.expansion) = 4
    #     self.fc_new = nn.Linear(512 * block.expansion, num_classes)
    #     # self.fc_regress = nn.Linear(512 * block.expansion, 1)

    #     ###FPN or seg 
        
    #     # Top layer
    #     self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

    #     # Smooth layers
    #     self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    #     # Lateral layers
    #     self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
    #     self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

	# 	# Semantic branch
    #     self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)
    #     # num_groups, num_channels
    #     self.gn1 = nn.GroupNorm(128, 128) 
    #     self.gn2 = nn.GroupNorm(256, 256)


    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    #     # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    #     if zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.bn3.weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.bn2.weight, 0)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)
    # def _upsample(self, x, h, w):
    #     return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    
    # def _upsample_add(self, x, y):
    #     '''Upsample and add two feature maps.
    #     Args:
    #       x: (Variable) top feature map to be upsampled.
    #       y: (Variable) lateral feature map.
    #     Returns:
    #       (Variable) added feature map.
    #     Note in PyTorch, when input size is odd, the upsampled feature map
    #     with `F.upsample(..., scale_factor=2, mode='nearest')`
    #     maybe not equal to the lateral feature map size.
    #     e.g.
    #     original input size: [N,_,15,15] ->
    #     conv2d feature map size: [N,_,8,8] ->
    #     upsampled feature map size: [N,_,16,16]
    #     So we choose bilinear upsample which supports arbitrary output sizes.
    #     '''
    #     _,_,H,W = y.size()
    #     return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     c1 = self.maxpool(x)

    #     c2 = self.layer1(c1)
    #     c3 = self.layer2(c2)
    #     c4 = self.layer3(c3)
    #     c5 = self.layer4(c4)
    #     fc = self.avgpool(c5)
    #     fc_ = torch.flatten(fc, 1)
    #     x_class = self.fc_new(fc_)
    #     # x_regress = self.fc_regress(fc_)
    #     # return x
    #     # add softmax
    #     # x = nn.Softmax(x)
    #     #scal_fpn 
    #     '''https://github.com/ElephantGit/SemanticSegmentationUsingFPN_PanopticFeaturePyramidNetworks/blob/master/model/FPN.py
    #     '''
        
    #     # Top-down
    #     p5 = self.toplayer(c5)  
    #     # Semantic
    #     _, _, h, w = c2.size()
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h//4, w//4)
    #     # 256->256
    #     s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h//2, w//2)
    #     # 256->128
    #     s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)
  
    #     seg_mask = self._upsample(self.conv3((s5)), 4 * h, 4 * w)
        
    #     return x_class,seg_mask #x_regress
 
 
    
if __name__ == "__main__":
    import os
    import time
     
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
     
    transform_input = False


    # model = res50_cls_res(transform_input=transform_input, block=Bottleneck, layers=[3, 4, 6, 3])
    # model = FPN_seg(transform_input=transform_input, block=Bottleneck, layers=[3, 4, 6, 3])
    # model = FPN_cls_seg(transform_input=transform_input, block=Bottleneck, layers=[3, 4, 6, 3])
    # model = FPT_cls_seg(transform_input=transform_input, cls_seg=False, block=Bottleneck, layers=[3, 4, 6, 3])
    model = FPT_cls_seg(transform_input=transform_input, cls_seg=True, block=Bottleneck, layers=[3, 4, 6, 3])

   
    model.eval()
    resolution = 256
    
    input = torch.randn(1, 3, resolution, resolution).cuda()
    model = model.cuda() 
    
    t0 = time.time()
    t=0
    for i in range(1200):
        
        torch.cuda.synchronize()
        t1 = time.time()
        outs = model(input)
        
        torch.cuda.synchronize()
        if i >199:
            t += time.time() - t1
    print(t/1000)
    
    
    # for out in outs:
    #     print(out.shape)
 
    # print(model)
    # from torchsummary import summary
    # summary(model, (3, resolution, resolution), batch_size=1,device="cpu")