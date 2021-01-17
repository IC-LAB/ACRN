"""
train code，refer to :https://github.com/felixgwu/img_classification_pk_pytorch
                      https://github.com/frederick0329/Image-Classification
resnet refer to : https://github.com/pytorch/vision/tree/master/torchvision/models

fpn refer to :
""" 
import os
import sys
# sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

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

import torch.distributed as dist
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel

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
                'FPN_seg':256, # 4卡
                'FPN_cls_seg':256,
              }
################ setting ####################################################### 
#  CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10001 main_distribute.py

args.resume = False # 从断点继续训练
args.resume_file = ''#"/home/tangrx/pytorch/gap/train_with_fpn_distribute/logs/camelyon512/with_0_5/FPN_cls/camelyon512_alldata_res50_fpn_stainnorm_w0_5_lr0.01_T60_mean0.5/model_last.pth.tar"
args.pretrained = False # 加载保存的模型
args.pt_path = "./model/resnet50_cls_camelyon512.pth.tar" # 加载模型路径

#=================================================
test_code = False
args.distribute = True

args.model_name = 'vgg16_cls' # vgg 改lr=0.01
args.transform_input = False  # false==>0.5
args.stainnorm = False

data_selcect = 'camelyon512' # paip512 , camelyon512 

args.gpu_number = '0,1,2,3' 

################################
args.gpus = len(args.gpu_number.split(','))

args.resize = True
args.resize_size = resize_dict[args.model_name]
#========================================================
args.cls = True if '_cls' in args.model_name else False
args.res = True if '_res' in args.model_name else False
args.seg = True if '_seg' in args.model_name else False 

if 'vgg' in args.model_name:
    args.lr = 0.01 # cls=0.1 , fpn=0.01  
else:
    args.lr = 0.1


args.nproc_per_node = args.gpus 
# args.local_rank = 0 
if args.distribute:
    args.batch_size= 128//args.gpus   # batch size = 40 ### total= 128
else:
    args.batch_size= 128  # batch size = 40 ### total= 128
args.num_workers = 2
args.epochs = 60

if args.stainnorm:
    if data_selcect == 'camelyon512':
        args.img_root = '../../../camelyon16/figure_512_tile_random_stainnorm' 
        logroot = './logs/camelyon512/with_0_5/'
    elif data_selcect == 'paip512':
        args.img_root = '../../../PAIP2019Challenge/figure_512_tile_stainnorm' 
        logroot = './logs/paip512/with_0_5/'
else:
    if data_selcect == 'camelyon512':
        args.img_root = '../../../camelyon16/figure_512_tile_random' 
        logroot = './logs/camelyon512/with_0_5/'
    elif data_selcect == 'paip512':
        args.img_root = '../../../PAIP2019Challenge/figure_512_tile' 
        logroot = './logs/paip512/with_0_5/'

if args.stainnorm:
    if args.transform_input:
        logname = data_selcect + '_' + args.model_name + '_stainnorm_w0_5_lr'+str(args.lr)+'_T'+str(args.epochs)+'_mean_resize'
    else:
        logname = data_selcect + '_' + args.model_name + '_stainnorm_w0_5_lr'+str(args.lr)+'_T'+str(args.epochs)+'_mean0.5_resize'
else:
    if args.transform_input:
        logname = data_selcect + '_' + args.model_name + '_w0_5_lr'+str(args.lr)+'_T'+str(args.epochs)+'_mean_resize'
    else:
        logname = data_selcect + '_' + args.model_name + '_w0_5_lr'+str(args.lr)+'_T'+str(args.epochs)+'_mean0.5_resize'   
args.log_dir = os.path.join(logroot, args.model_name, logname) 

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

data_root = args.img_root+ '/dataframe_stainnorm_addroot/'  
args.train_data_dir = data_root + 'train.csv' 
args.val_data_dir = data_root +'val.csv' 
args.test_data_dir = data_root + 'test.csv'  

if test_code:    
    args.train_data_dir =  './test_code.csv'
    args.val_data_dir   =  './test_code.csv'
    args.test_data_dir  =  './test_code.csv'

######################################################################
args.arch = args.model_name
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_number) 
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader): 
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
def data(train_data_dir,val_data_dir, test_data_dir,batch_size, transform ,num_workers):
    ''' 
    using MyDataset,DataLoader
    return trainlodaer, testloader
    '''
    # train_records = pd.read_csv(train_data_dir,header=0)[:100]
    # val_records = pd.read_csv(val_data_dir,header=0)[:100]
    # test_records = pd.read_csv( test_data_dir,header=0)[:100]
    train_records = pd.read_csv(train_data_dir,header=0)#[:100]
    val_records = pd.read_csv(val_data_dir,header=0)#[:100]
    test_records = pd.read_csv( test_data_dir,header=0)#[:100]

    trainset = MyDataset( dataframe= train_records,dataset='train',loader='pil_loader', data_transforms = transform, args=args)
    # trainloader = DataLoaderX(trainset, batch_size= batch_size, shuffle=True, num_workers= num_workers)
    valset = MyDataset( dataframe= val_records,dataset='test',loader='pil_loader', data_transforms = transform, args=args)
    # valloader = DataLoaderX(valset, batch_size= batch_size, shuffle=False, num_workers= num_workers)
    testset = MyDataset( dataframe=test_records,dataset='test',loader='pil_loader', data_transforms = transform, args=args)
    # testloader = DataLoaderX(testset, batch_size= batch_size, shuffle=False, num_workers= num_workers)
    
    assert train_records['img_path'][1].split("/")[-3]== 'patches' or train_records['img_path'][1].split("/")[-4] == 'patches', "filename error"
    assert val_records['img_path'][1].split("/")[-3]== 'patches' or val_records['img_path'][1].split("/")[-4] == 'patches', "filename error"
    assert test_records['img_path'][1].split("/")[-3]== 'patches' or test_records['img_path'][1].split("/")[-4] == 'patches', "filename error"
    
    if args.local_rank ==0:
        print("train data",len(train_records)) 
        print(train_records)
        print("val data",len(val_records)) 
        print(val_records)
        print("test data",len(test_records)) 
        print(test_records)
        print("dataloading and start training")
    # return trainloader, valloader, testloader
    return trainset, valset, testset

def save_checkpoint(state, save_dir,filename='checkpoint.pth.tar'):  
    torch.save(state, os.path.join(save_dir, filename)) 
 
def main():
    #==================== prepare dataloader========================= 
    if args.transform_input:
        print('not transform input, (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)')
        transform = {'train':transforms.Compose([transforms.ToTensor() ]), 'test':transforms.Compose([transforms.ToTensor() ])  } #(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    else:
        print('transform input (mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)')
        transform = {'train':transforms.Compose([transforms.ToTensor(),#]),
                                            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5) )]), #(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                'test':transforms.Compose([transforms.ToTensor(), #])
                                           transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5) )])  # (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                }
    train_dataset,val_dataset,test_dataset = data(train_data_dir=args.train_data_dir,
                                                val_data_dir=args.val_data_dir,
                                                test_data_dir=args.test_data_dir,
                                                batch_size= args.batch_size,
                                                transform = transform,
                                                num_workers = args.num_workers)

    if args.local_rank==0:
        # log_print('total gpu: %d, gpu using ids: %s (reference id)'%(torch.cuda.device_count(), args.gpu_ids))
        f_log = open(os.path.join(args.log_dir, 'log.txt'), 'w')
        def log_print(*args):
            print(*args)
            print(*args, file=f_log)
        log_print('total gpu: %d '%(torch.cuda.device_count() ))
    
    if args.gpus>0 and args.distribute:
        torch.distributed.init_process_group('nccl', init_method='env://') #nccl,gloo, 'tcp://127.0.0.1:1111'
        # torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:10001',world_size=4,rank=args.local_rank) #nccl,gloo, , world_size=args.world_size, rank=args.local_rank  
        assert torch.distributed.is_initialized(), 'torch distribute initial fail'
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
  
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=args.local_rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.gpus, rank=args.local_rank, shuffle=False )
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=args.gpus , rank=args.local_rank , shuffle=False)

        trainloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=True,  sampler=train_sampler, pin_memory = True)       
        valloader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=False, sampler = val_sampler, pin_memory = True)
        testloader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=False, sampler = test_sampler, pin_memory = True)
    else:
        trainloader = DataLoaderX(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers= args.num_workers)
        valloader = DataLoaderX(val_dataset, batch_size= args.batch_size, shuffle=False, num_workers= args.num_workers)
        testloader = DataLoaderX(test_dataset, batch_size= args.batch_size, shuffle=False, num_workers= args.num_workers)
        
    #=======================define model=================================
    # model = resnet50(pretrained=args.pretrained, num_classes=2)  
    model = get_model(model_name=args.model_name,transform_input=args.transform_input)
    if args.distribute:
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device = args.local_rank, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define optimizer  
    optimizer = get_optimizer(model, args) 
    
    # loading model #initial
    start_epoch = 1
    if not args.resume and args.pretrained: # 加载保存的模型
        print('loading model from: ', args.pt_path) 
        checkpoint_class = torch.load(args.pt_path, map_location=torch.device('cpu'))   
        pretrained_dict = checkpoint_class['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        start_epoch = 1
    elif args.resume: # 恢复上次的训练状态 
        print("Resume from checkpoint...", args.resume_file)
        checkpoint = torch.load(args.resume_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1 
    else:
        print('initial model and no loading pre-model')

    cudnn.benchmark = True  
 
    # define loss function (criterion) and optimizer 
    criterion_cls = nn.CrossEntropyLoss().cuda()  
    criterion_seg = nn.CrossEntropyLoss().cuda()  
    criterion_res = nn.MSELoss().cuda()
    
    # set random seed
    torch.manual_seed(args.seed)
    trainer = Trainer(model, criterion_cls, criterion_res, criterion_seg, optimizer, args)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer,   T_max=50, eta_min=4e-08, last_epoch=-1) # 余弦退火
 
    # =================================tensorboard==========================
    if args.distribute:
        if args.local_rank==0:
            configure(args.log_dir, flush_secs=5) 
            f_log = open(os.path.join(args.log_dir, 'log.txt'), 'w')
            def log_print(*args):
                print(*args)
                print(*args, file=f_log)
            log_print('args:')
            log_print(args) 
    else:
        configure(args.log_dir, flush_secs=5) 
        f_log = open(os.path.join(args.log_dir, 'log.txt'), 'w')
        def log_print(*args):
            print(*args)
            print(*args, file=f_log)
        log_print('args:')
        log_print(args) 
    
    #==================================================== train ===========================================================
    best_acc = 90 
    for epoch in range(start_epoch, 1 + args.epochs):   
        train_loss,train_loss_cls,train_loss_res,train_loss_seg, lr, optimizer = trainer.train(trainloader, epoch)  

        if args.distribute:
            dist.all_reduce(train_loss)
            train_loss = train_loss/args.gpus
            if args.cls:
                dist.all_reduce(train_loss_cls)
                train_loss_cls = train_loss_cls/args.gpus
            if args.res:
                dist.all_reduce(train_loss_res) 
                train_loss_res = train_loss_res/args.gpus
            if args.seg:
                dist.all_reduce(train_loss_seg)
                train_loss_seg = train_loss_seg/args.gpus
        if args.local_rank==0:
            log_print('epoch ', epoch)
            train_loss = train_loss.item()
            if args.cls:
                train_loss_cls =  train_loss_cls.item()
            if args.res:
                train_loss_res = train_loss_res.item()
            if args.seg:
                train_loss_seg = train_loss_seg.item() 
            if args.cls and not args.res and args.seg: # cls + seg
                log_print('TRAIN:lr{lr}, loss{loss:.4f}, loss_cls{loss:.4f}, loss_seg{loss:.4f}'.format(
                         lr=lr,loss=train_loss,loss_cls=train_loss_cls, loss_seg=train_loss_seg ))
            elif args.cls and args.res and not args.seg: # cls + res
                log_print('TRAIN:lr{lr}, loss{loss:.4f}, loss_cls{loss:.4f}, loss_res{loss:.4f}'.format(
                         lr=lr,loss=train_loss,loss_cls=train_loss_cls, loss_res=train_loss_res ))
            elif args.cls and not args.res and not args.seg: # cls
                log_print('TRAIN:lr{lr}, loss{loss:.4f},'.format(lr=lr,loss=train_loss ))
            elif not args.cls and not args.res and args.seg: # seg
                log_print('TRAIN:lr{lr}, loss{loss:.4f}, loss_seg{loss:.4f}'.format(
                         lr=lr,loss=train_loss, loss_seg=train_loss_seg ))
            else:
                log_print('cls res seg error')
 
        # lr_scheduler.step() # 余弦退火法
        #================================================validate ==================================================== 
        val_loss,val_loss_cls, val_loss_res, val_loss_seg, \
            val_cls_acc,val_seg_cls_acc,val_seg_acc,val_seg_mIoU = trainer.test(valloader, epoch) 

        if args.distribute:
            dist.all_reduce(val_loss)
            val_loss = val_loss/args.gpus
            if args.cls:  
                dist.all_reduce(val_loss_cls), dist.all_reduce(val_cls_acc)
                val_loss_cls = val_loss_cls/args.gpus
                val_cls_acc = val_cls_acc/args.gpus
            if args.res:
                dist.all_reduce(val_loss_res)
                val_loss_res = val_loss_res/args.gpus
            if args.seg:
                dist.all_reduce(val_loss_seg), dist.all_reduce(val_seg_cls_acc), dist.all_reduce(val_seg_acc), dist.all_reduce(val_seg_mIoU)
                val_loss_seg = val_loss_seg/args.gpus
                val_seg_cls_acc, val_seg_acc, val_seg_mIoU =  val_seg_cls_acc/args.gpus, val_seg_acc/args.gpus, val_seg_mIoU/args.gpus 
        if args.local_rank==0:
            val_loss = val_loss.item()
            if args.cls:
                val_loss_cls = val_loss_cls.item()
                val_cls_acc = val_cls_acc.item()
            if args.res:
                val_loss_res = val_loss_res.item()
            if args.seg:
                val_loss_seg = val_loss_seg.item()
                val_seg_cls_acc, val_seg_acc, val_seg_mIoU = val_seg_cls_acc.item(), val_seg_acc.item(), val_seg_mIoU.item()
            if args.cls and not args.res and args.seg: # cls + seg
                log_print('VAL:lr{lr}, loss{loss:.4f}, loss_cls{loss_cls:.4f}, loss_seg{loss_seg:.4f}, \
                          cls_acc{cls_acc:.4f},seg_cls_acc{seg_cls_acc:.4f},seg_acc{seg_acc:.4f}'\
                         .format(lr=lr, loss=val_loss, loss_cls=val_loss_cls, loss_seg=val_loss_seg,
                         cls_acc=val_cls_acc, seg_cls_acc=val_seg_cls_acc, seg_acc=val_seg_acc))
            elif args.cls and args.res and not args.seg: # cls + res:
                log_print('VAL:lr{lr}, loss{loss:.4f}, val_cls_acc{class_acc:.4f}, val_mae{val_loss_res:.4f}'\
                         .format(lr=lr,loss=val_loss, class_acc=val_cls_acc, val_loss_res = val_loss_res))
            elif args.cls and not args.res and not args.seg: # cls
                log_print('VAL:lr{lr}, loss{loss:.4f}, val_cls_acc{class_acc:.4f}'\
                          .format(lr=lr,loss=val_loss, class_acc=val_cls_acc))
            elif not args.cls and not args.res and args.seg: # seg
                log_print('VAL:lr{lr}, loss{loss:.4f}, seg_cls_acc{class_acc:.4f},seg_acc{seg_acc:.4f}'\
                         .format(lr=lr,loss=val_loss, class_acc=val_seg_cls_acc,seg_acc=val_seg_acc))
            else:
                log_print('cls res seg error') 

            # tensorboard
            log_value('lr', lr, epoch)
            log_value('train_loss_total', train_loss, epoch)
            log_value('val_loss_total', val_loss, epoch)
            if args.cls:
                log_value('train_loss_cls', train_loss_cls, epoch)
                log_value('val_loss_cls', val_loss_cls, epoch)
                log_value('val_cls_acc', val_cls_acc, epoch)   
            if args.res:
                log_value('val_loss_res', val_loss_res, epoch)    
            if args.seg:
                log_value('train_loss_seg', train_loss_seg, epoch) 
                log_value('val_loss_seg', val_loss_seg, epoch)  
                log_value('val_seg_cls_acc', val_seg_cls_acc, epoch)  
                log_value('val_seg_acc', val_seg_acc, epoch) 
                log_value('val_seg_mIoU', val_seg_mIoU, epoch)  
    
            save_checkpoint({ 'args': args, 'epoch': epoch, 'state_dict': model.state_dict(),  'optimizer_state_dict': optimizer.state_dict(),
                                 }, args.log_dir,filename='model_last.pth.tar') 
    
        #=========================================testing===========================================
        if args.seg:
            is_best = val_seg_acc > best_acc
        else:
            is_best = val_cls_acc > best_acc
        if is_best: 
            best_acc = val_seg_acc if args.seg else val_cls_acc
            test_loss,test_loss_cls,test_loss_res, test_loss_seg, \
                test_cls_acc,test_seg_cls_acc,test_seg_acc,test_seg_mIoU = trainer.test(testloader, epoch)
 
            if args.distribute:
                dist.all_reduce(test_loss)
                test_loss = test_loss/args.gpus
                if args.cls:
                    dist.all_reduce(test_loss_cls), dist.all_reduce(test_cls_acc)
                    test_loss_cls = test_loss_cls/args.gpus
                    test_cls_acc = test_cls_acc/args.gpus
                if args.res:
                    dist.all_reduce(test_loss_res)
                    test_loss_res = test_loss_res/args.gpus
                if args.seg:
                    dist.all_reduce(test_loss_seg), dist.all_reduce(test_seg_cls_acc), dist.all_reduce(test_seg_acc), dist.all_reduce(test_seg_mIoU)  
                    test_loss_seg = test_loss_seg/args.gpus
                    test_seg_cls_acc, test_seg_acc, test_seg_mIoU =  test_seg_cls_acc/args.gpus, test_seg_acc/args.gpus, test_seg_mIoU/args.gpus 
            
            best_epoch = epoch    
            if args.local_rank==0:
                test_loss = test_loss.item()
                if args.cls:
                    test_loss_cls = test_loss_cls.item()
                    test_cls_acc = test_cls_acc.item()
                if args.res:
                    test_loss_res = test_loss_res.item()
                if args.seg:
                    test_loss_seg = test_loss_seg.item()
                    test_seg_cls_acc, test_seg_acc, test_seg_mIoU = test_seg_cls_acc.item(), test_seg_acc.item(), test_seg_mIoU.item()
                print(Fore.RED)
                if args.cls and not args.res and args.seg: # cls + seg
                    log_print('test:lr{lr}, loss{loss:.4f}, loss_cls{loss_cls:.4f}, loss_seg{loss_seg:.4f}, \
                            cls_acc{cls_acc:.4f},seg_cls_acc{seg_cls_acc:.4f},seg_acc{seg_acc:.4f}'\
                            .format(lr=lr, loss=test_loss, loss_cls=test_loss_cls, loss_seg=test_loss_seg,
                            cls_acc=test_cls_acc, seg_cls_acc=test_seg_cls_acc, seg_acc=test_seg_acc))
                elif args.cls and args.res and not args.seg: # cls + res:
                    log_print('test:lr{lr}, loss{loss:.4f}, test_cls_acc{class_acc:.4f}, test_mae{test_loss_res:.4f}'\
                            .format(lr=lr,loss=test_loss, class_acc=test_cls_acc, test_loss_res = test_loss_res))
                elif args.cls and not args.res and not args.seg: # cls
                    log_print('test:lr{lr}, loss{loss:.4f}, test_cls_acc{class_acc:.4f}'\
                            .format(lr=lr,loss=test_loss, class_acc=test_cls_acc))
                elif not args.cls and not args.res and args.seg: # seg
                    print(Fore.RED + 'test:lr{lr}, loss{loss:.4f}, seg_cls_acc{class_acc:.4f},seg_acc{seg_acc:.4f}'\
                            .format(lr=lr,loss=test_loss, class_acc=test_seg_cls_acc,seg_acc=test_seg_acc))
                else:
                    log_print('cls res seg error') 
                print(Fore.RESET)
                save_checkpoint({ 'args': args, 'epoch': epoch, 'best_epoch': best_epoch,   'state_dict': model.state_dict(), 'best_acc': best_acc, 
                                 }, args.log_dir,filename='model_best.pth.tar') 
   
    # ===========保存最后epoch的模型================
    if args.local_rank==0:
        save_checkpoint({ 'args': args, 'epoch': epoch, 'best_epoch': best_epoch,  'state_dict': model.state_dict()
                        }, args.log_dir,filename='model_end.pth.tar')

    test_loss,test_loss_cls,test_loss_res, test_loss_seg, \
                test_cls_acc,test_seg_cls_acc,test_seg_acc,test_seg_mIoU = trainer.test(testloader, epoch)
    if args.distribute:
        dist.all_reduce(test_loss)
        test_loss = test_loss/args.gpus
        if args.cls:
            dist.all_reduce(test_loss_cls), dist.all_reduce(test_cls_acc)
            test_loss_cls = test_loss_cls/args.gpus
            test_cls_acc = test_cls_acc/args.gpus
        if args.res:
            dist.all_reduce(test_loss_res)
            test_loss_res = test_loss_res/args.gpus
        if args.seg:
            dist.all_reduce(test_loss_seg), dist.all_reduce(test_seg_cls_acc), dist.all_reduce(test_seg_acc), dist.all_reduce(test_seg_mIoU)  
            test_loss_seg = test_loss_seg/args.gpus
            test_seg_cls_acc, test_seg_acc, test_seg_mIoU =  test_seg_cls_acc/args.gpus, test_seg_acc/args.gpus, test_seg_mIoU/args.gpus 
     
    if args.local_rank==0:
        test_loss = test_loss.item()
        if args.cls:
            test_loss_cls = test_loss_cls.item()
            test_cls_acc = test_cls_acc.item()
        if args.res:
            test_loss_res = test_loss_res.item()
        if args.seg:
            test_loss_seg = test_loss_seg.item()
            test_seg_cls_acc, test_seg_acc, test_seg_mIoU = test_seg_cls_acc.item(), test_seg_acc.item(), test_seg_mIoU.item()
        print(Fore.GREEN)
        if args.cls and not args.res and args.seg: # cls + seg
            log_print('test:lr{lr}, loss{loss:.4f}, loss_cls{loss_cls:.4f}, loss_seg{loss_seg:.4f}, \
                    cls_acc{cls_acc:.4f},seg_cls_acc{seg_cls_acc:.4f},seg_acc{seg_acc:.4f}'\
                    .format(lr=lr, loss=test_loss, loss_cls=test_loss_cls, loss_seg=test_loss_seg,
                    cls_acc=test_cls_acc, seg_cls_acc=test_seg_cls_acc, seg_acc=test_seg_acc))
        elif args.cls and args.res and not args.seg: # cls + res:
            log_print('test:lr{lr}, loss{loss:.4f}, test_cls_acc{class_acc:.4f}, test_mae{test_loss_res:.4f}'\
                    .format(lr=lr,loss=test_loss, class_acc=test_cls_acc, test_loss_res = test_loss_res))
        elif args.cls and not args.res and not args.seg: # cls
            log_print('test:lr{lr}, loss{loss:.4f}, test_cls_acc{class_acc:.4f}'\
                    .format(lr=lr,loss=test_loss, class_acc=test_cls_acc))
        elif not args.cls and not args.res and args.seg: # seg
            print(Fore.RED + 'test:lr{lr}, loss{loss:.4f}, seg_cls_acc{class_acc:.4f},seg_acc{seg_acc:.4f}'\
                    .format(lr=lr,loss=test_loss, class_acc=test_seg_cls_acc,seg_acc=test_seg_acc))
        else:
            log_print('cls res seg error') 

        print(Fore.RESET)
        
if __name__ == "__main__":
    main()
    
