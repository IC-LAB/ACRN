'''
测试网络模型，单独一张图片，保存在dataframe中，并输出精度，还有融合模型预测的精度。
测试分割网络模型，保存图像。
'''
import os
import sys
# sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil
from colorama import Fore

import torch
from torchsummary import summary
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from colorama import Fore
from tensorboard_logger import configure, log_value

from resnet import *  
from trainer import Trainer,get_optimizer
from dataloader import MyDataset
import glob 

import torch.distributed as dist
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel
from multiprocessing import Process
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader): 
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

from args import arg_parser, arch_resume_names
global args 
args = arg_parser.parse_args()  

resize_dict = {'res18_cls_res': 256,  # 0.5 卡
                'res18_cls':256,
                'res50_cls_res':256, # 2 卡
                'res50_cls':256,
                'vgg16_cls_res':256,  # >4卡 lr=0.1 梯度爆炸
                'vgg16_cls':256,      # 4卡
                'vgg19_cls_res':256,  
                'vgg19_cls':256,
                'inception_cls_res':299, #  2 卡 
                'inception_cls':299,
                'FPN':256, # 4卡
                'FPN_cls':256,
              }
model_dt = {'res18_cls_res': 'cls_res',  # 0.5 卡
            'res18_cls':'cls',
            'res50_cls_res':'cls_res', # 2 卡
            'res50_cls':'cls',
            'vgg16_cls_res':'cls_res',  # >4卡 lr=0.1 梯度爆炸
            'vgg16_cls':'cls',   # 4卡
            'vgg19_cls_res':'cls_res',  
            'vgg19_cls':'cls',
            'inception_cls_res':'cls_res', #  2 卡 
            'inception_cls':'cls',
            'FPN_seg':'seg', # 4卡
            'FPN_cls_seg':'cls_seg',
            }
args.resize = True
args.resize_size = resize_dict[args.model_name]
args.batch_size=128

# def test():
def inference(gpu_number=0, model_path='',model_name='',test_data_dir='', infer=True, test=False, df_save_dir='', args=args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)  
    
    if not model_name:
        model_name = model_path.split('/')[-3]
    args.cls = True if 'cls' in model_dt[model_name] else False
    args.res = True if 'res' in model_dt[model_name] else False
    args.seg = True if 'seg' in model_dt[model_name] else False 

    if 'mean0.5' in model_path:
        args.transform_input = False # False==>0.5
        transform = { 'test':transforms.Compose([transforms.ToTensor(),  transforms.Normalize( mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) ])  }    
    else:
        args.transform_input = True # False==>0.5
        transform = { 'test':transforms.Compose([transforms.ToTensor()]) }    

    if not test_data_dir:
        if 'stainnorm' in model_path:
            if 'camelyon' in model_path:
                args.img_root = '../../../camelyon16/figure_512_tile_random_stainnorm'  
            else :
                args.img_root = '../../../PAIP2019Challenge/figure_512_tile_stainnorm'  
        else:
            if 'camelyon' in model_path: 
                args.img_root = '../../../camelyon16/figure_512_tile_random'  
            else :
                args.img_root = '../../../PAIP2019Challenge/figure_512_tile'   
        test_data_dir = args.img_root + '/dataframe_stainnorm_addroot/val/val_with_0_5.csv'  
    else:
        args.img_root = ''   

    if not df_save_dir:
        df_save_dir = '/'.join(model_path.split('/')[:-1]) + '/predict.csv'
    
    print(Fore.GREEN)
    print(model_name)
    print(Fore.RESET)
    print(model_path)

    model = get_model(model_name = model_name, transform_input = args.transform_input)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = get_optimizer(model, args) 
 
    model.load_state_dict(torch.load(model_path)['state_dict']) 

    cudnn.benchmark = True  
    
    # define loss function (criterion) and optimizer 
    criterion_cls = nn.CrossEntropyLoss().cuda()  
    criterion_seg = nn.CrossEntropyLoss().cuda()  
    criterion_res = nn.MSELoss().cuda()

    # set random seed
    torch.manual_seed(args.seed)
    trainer = Trainer(model, criterion_cls, criterion_res, criterion_seg, optimizer, args)
    
    df = pd.read_csv( test_data_dir,header=0)#[:100]
    
    # #del
    # for i in range(0,np.shape(df)[0]):  
    #     # ==========================数据处理=========================
    #     img_path = args.img_root + df.loc[i,'img_path']
    
    # =============================单张图像测试并保存dataframe=========
    if infer:   
        testset = MyDataset( dataframe=df,dataset='test',loader='pil_loader', data_transforms = transform, args=args)
        # assert testset['img_path'][1].split("/")[-3]== 'patches' or testset['img_path'][1].split("/")[-4] == 'patches', "filename error"
        testloader = DataLoaderX(testset, batch_size=args.batch_size, shuffle=False, num_workers= 1)
        
        if args.cls and not args.res:
            df['model_cls_class_0'] = None  
            df['model_cls_class_1'] = None 
        if args.res:
            df['model_cls_res_regress_1'] = None 
            df['model_cls_res_class_0'] = None 
            df['model_cls_res_class_1'] = None  
            
        trainer.inference(df,testloader, transform, df_save_dir) 

    #=======================测试精度=========================
    if test:
        testset = MyDataset( dataframe=df,dataset='test',loader='pil_loader', data_transforms = transform, args=args)
        # assert testset['img_path'][1].split("/")[-3]== 'patches' or testset['img_path'][1].split("/")[-4] == 'patches', "filename error"
        testloader = DataLoaderX(testset, batch_size= args.batch_size, shuffle=False, num_workers= args.num_workers)
        
        test_loss,test_loss_cls,test_loss_res, test_loss_seg, \
            test_cls_acc,test_seg_cls_acc,test_seg_acc,test_seg_mIoU = trainer.test(testloader, epoch=1)

        test_loss = test_loss.item()
        if args.cls:
            test_loss_cls = test_loss_cls.item()
            test_cls_acc = test_cls_acc.item()
        if args.res:
            test_loss_res = test_loss_res.item()
        if args.seg:
            test_loss_seg = test_loss_seg.item()
            test_seg_cls_acc, test_seg_acc, test_seg_mIoU = test_seg_cls_acc.item(), test_seg_acc.item(), test_seg_mIoU.item()

        if args.cls and not args.res and args.seg: # cls + seg
            print('test:lr{lr}, loss{loss:.4f}, loss_cls{loss_cls:.4f}, loss_seg{loss_seg:.4f}, \
                    cls_acc{cls_acc:.4f},seg_cls_acc{seg_cls_acc:.4f},seg_acc{seg_acc:.4f}'\
                    .format(lr=lr, loss=test_loss, loss_cls=test_loss_cls, loss_seg=test_loss_seg,
                    cls_acc=test_cls_acc, seg_cls_acc=test_seg_cls_acc, seg_acc=test_seg_acc))
        elif args.cls and args.res and not args.seg: # cls + res:
            print('test: loss{loss:.4f}, test_cls_acc{class_acc:.4f}, test_mae{test_loss_res:.4f}'\
                    .format(loss=test_loss, class_acc=test_cls_acc, test_loss_res = test_loss_res))
        elif args.cls and not args.res and not args.seg: # cls
            print('test: loss{loss:.4f}, test_cls_acc{class_acc:.4f}'\
                    .format(loss=test_loss, class_acc=test_cls_acc))
        elif not args.cls and not args.res and args.seg: # seg
            print( 'test: loss{loss:.4f}, seg_cls_acc{class_acc:.4f},seg_acc{seg_acc:.4f}'\
                    .format(loss=test_loss, class_acc=test_seg_cls_acc,seg_acc=test_seg_acc))
        
 
if __name__ == "__main__":   
    
    # args.batch_size = 1
    # model_test = './logs/paip512/with_0_5/stainnorm_mean0.5/*/*/*/model_best.pth.tar'
    # model_paths = glob.glob(model_test)
    # for model_path in model_paths:
    #     inference(gpu_number=8, model_path=model_path,test_data_dir='./test_infer.csv',df_save_dir='./test_infer_result.csv')
    #     break

    # args.batch_size = 10
    # model_test = './logs/paip512/with_0_5/stainnorm_mean0.5/*/*/*/model_best.pth.tar'
    # model_paths = glob.glob(model_test)
    # for model_path in model_paths:
    #     inference(gpu_number=8, model_path=model_path,test_data_dir='./test_infer.csv',df_save_dir='./test_infer_result1.csv')
    #     break
     

    # model_1 = './logs/camelyon512/with_0_5/stainnorm_mean/*/*/*/model_best.pth.tar'
    # model_paths = glob.glob(model_1)
    # for model_path in model_paths:
    #     inference(gpu_number=8, model_path=model_path)

    # model_2 = './logs/camelyon512/with_0_5/stainnorm_mean0.5/*/*/*/model_best.pth.tar'
    # model_paths = glob.glob(model_2)
    # for model_path in model_paths:
    #     inference(gpu_number=8, model_path=model_path)

    model_3 = './logs/paip512/with_0_5/stainnorm_mean/*/*/*/model_best.pth.tar'
    model_paths = glob.glob(model_3)
    for model_path in model_paths:
        inference(gpu_number=7, model_path=model_path)

    model_4 = './logs/paip512/with_0_5/stainnorm_mean0.5/*/*/*/model_best.pth.tar'
    model_paths = glob.glob(model_4)
    for model_path in model_paths:
        inference(gpu_number=7, model_path=model_path)
