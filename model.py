from resnet import *
from vgg import *
from inception import *
from Botnet import *
from deeplabv3.deeplabv3 import get_deeplab
from cls_T_seg import get_cls_T_seg

def get_model(model_name = 'resnet50',transform_input=False, pretrained=True, block=Bottleneck, progress=True,resolution=(256,256),img_root='', **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    # FPN , FPN_cls, res50_cls_res, res50_cls, res18_cls, res18_cls_res
    """
    model_path_pretrained = {
        'res18_cls_res':'./model/resnet18.pth',
        'res18_cls':'./model/resnet18.pth',
        'res50_cls_res':'./model/resnet50.pth',
        'res50_cls':'./model/resnet50.pth',
        'vgg16_cls_res':'./model/vgg16.pth',
        'vgg16_cls':'./model/vgg16.pth',
        'vgg19_cls_res':'./model/vgg19.pth',
        'vgg19_cls':'./model/vgg19.pth',
        'inception_cls_res':'./model/inception_v3.pth',
        'inception_cls':'./model/inception_v3.pth',

        'Botnet_cls':'./model/resnet50.pth',
        'Botnet_cls_res':'./model/resnet50.pth',

        'FPN_seg':'./model/resnet50.pth',   #
        'FPN_cls_seg':'./model/resnet50.pth',#

        'Botnet_seg':'./model/resnet50.pth',#
        'Botnet_cls_seg':'./model/resnet50.pth',#

        'FPT_seg': './model/resnet50.pth',#
        'FPT_cls_seg':'./model/resnet50.pth',#

        'FPT_Botnet_seg':'./model/resnet50.pth',#
        'FPT_Botnet_cls_seg': './model/resnet50.pth',#
        
        'deeplab_50_seg':"./deeplabv3/resnet/resnet50-19c8e357.pth",
        'deeplab_101_seg':"./deeplabv3/resnet/resnet101-5d3b4d8f.pth",

        'cls_T_seg':'./model/resnet50.pth',#
        'cls_T_seg_S':'./model/resnet50.pth',#

    }
    
    #============cls+res===========================
    
    if model_name == 'res18_cls_res':
        model = res50_cls_res(transform_input=transform_input, cls_res=True, block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    elif model_name == 'res50_cls_res':
        model = res50_cls_res(transform_input=transform_input, cls_res=True, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    elif model_name == 'vgg16_cls_res':
        model = vgg16_cls_res(transform_input=transform_input, cls_res=True)
    elif model_name == 'vgg19_cls_res':
        model = vgg19_cls_res(transform_input=transform_input, cls_res=True)
    elif model_name == 'inception_cls_res':
        model = inception_v3(transform_input=transform_input, cls_res=True)

    #============cls===========================
    elif model_name == 'res18_cls':
        model = res50_cls_res(transform_input=transform_input, cls_res=False, block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    elif model_name == 'res50_cls':
        model = res50_cls_res(transform_input=transform_input, cls_res=False, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    elif model_name == 'vgg16_cls':
        model = vgg16_cls_res(transform_input=transform_input, cls_res=False)
    elif model_name == 'vgg19_cls':
        model = vgg19_cls_res(transform_input=transform_input, cls_res=False)
    elif model_name == 'inception_cls':
        model = inception_v3(transform_input=transform_input, cls_res=False)

    #============  FPN  ===========================
    elif model_name == 'FPN_seg':
        model = FPN_seg(transform_input=transform_input, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs) 
    elif model_name == 'FPN_cls_seg':
        model = FPN_cls_seg(transform_input=transform_input, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs) 
    #============  FPT  ===========================
    elif model_name == 'FPT_seg':
        model = FPT_cls_seg(transform_input=transform_input, cls_seg=False, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs) 
    elif model_name == 'FPT_cls_seg':
        model = FPT_cls_seg(transform_input=transform_input, cls_seg=True, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs) 

    #==================Botnet==============
    elif model_name=='Botnet_cls':
        model = Botnet_cls_res(transform_input=transform_input, cls_res=False, block=Bottleneck_T, layers=[3, 4, 6, 3], resolution=resolution, **kwargs)
    elif model_name=='Botnet_cls_res':
        model = Botnet_cls_res(transform_input=transform_input, cls_res=True, block=Bottleneck_T, layers=[3, 4, 6, 3], resolution=resolution, **kwargs)

    elif model_name=='Botnet_seg':
        model = Botnet_cls_seg(transform_input=transform_input, cls_seg=False, block=Bottleneck_T, layers=[3, 4, 6, 3], resolution=resolution, **kwargs)
    elif model_name=='Botnet_cls_seg':
        model = Botnet_cls_seg(transform_input=transform_input, cls_seg=True, block=Bottleneck_T, layers=[3, 4, 6, 3], resolution=resolution, **kwargs)
    
    #====================FPT=====================
    elif model_name=='FPT_seg':
        model = FPT_cls_seg(transform_input=transform_input, cls_seg=False, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    elif model_name=='FPT_cls_seg':
        model = FPT_cls_seg(transform_input=transform_input, cls_seg=True, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
        

    #====================FPT Botnet=============
    elif model_name=='FPT_Botnet_seg':
        model = FPT_Botnet_cls_seg(transform_input=transform_input, cls_seg=False, block=Bottleneck_T, layers=[3, 4, 6, 3], resolution=resolution, **kwargs)
    elif model_name=='FPT_Botnet_cls_seg':
        model = FPT_Botnet_cls_seg(transform_input=transform_input, cls_seg=True, block=Bottleneck_T, layers=[3, 4, 6, 3], resolution=resolution, **kwargs)

    # ===========deeplab=========================
    elif model_name=='deeplab_50_seg':
        model = get_deeplab(transform_input=False, cls_seg=False,resnet ='ResNet50_OS16') 
    elif model_name=='deeplab_101_seg':
        model = get_deeplab(transform_input=False, cls_seg=False,resnet ='ResNet101_OS16') 

    # ===========cls_T_seg=========================
    elif model_name=='cls_T_seg':
        model = get_cls_T_seg(model_name='cls_T_seg_S', transform_input=transform_input, cls_seg=True, layers=[3, 4, 6, 3], **kwargs)
    elif model_name=='cls_T_seg_S':
        model = get_cls_T_seg(model_name='cls_T_seg_S', transform_input=transform_input, cls_seg=True, layers=[3, 4, 6, 3], **kwargs)

    if pretrained: # 加载下载好的模型参数
        # print("loading pretrained model from url")
        # model_dict = model.state_dict()
        # pretrained_dict = load_state_dict_from_url(model_urls[model_name], progress=progress)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict) 
        # model.load_state_dict(model_dict)
    
        # print('model_dict ',model_dict.keys() ) 
        # print('pretrained_dict',pretrained_dict.keys())
        # print('update_dict',update_dict.keys())
        if 'FPT_Botnet' in model_name:
            pt_path = './model/Botnet50_cls_paip256.pth.tar' if 'paip' in img_root else './model/Botnet50_cls_camelyon256.pth.tar'
            print('loading model from: ', pt_path) 
            checkpoint_class = torch.load(pt_path, map_location=torch.device('cpu'))   
            pretrained_dict = checkpoint_class['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        else:
            print('loading model from path: ', model_path_pretrained[model_name]) 
            pretrained_dict = torch.load(model_path_pretrained[model_name])   # , map_location=torch.device('cpu') 
            model_dict = model.state_dict()
            # print(model_dict.keys())
            update_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
            model_dict.update(update_dict)

    return model 
    
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
 
    # model = get_model('Botnet_cls_seg')

    # print(model)
    
    # model = get_deeplab(transform_input=False, cls_seg=False,resnet ='ResNet50_OS16')
    # model = get_cls_T_seg(transform_input=False, cls_seg=True)
    model = get_model(model_name = 'inception_cls',transform_input=False, pretrained=True)
    # model = get_model(model_name = 'FPN_cls_seg',transform_input=False, pretrained=True)

    # model.eval()
    # resolution = 299
    # input = torch.randn(2, 3, resolution, resolution) 
    # outs = model(input) 
    # for out in outs:
    #     print(out.shape)
 
    # print(model)
    # from torchsummary import summary
    # summary(model, (3, resolution, resolution), batch_size=1,device="cpu")


    model.eval()
    resolution = 299
    import time
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