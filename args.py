import os
import glob
import time
import argparse
import config 
arg_parser = argparse.ArgumentParser(description='Image classification PK main script')

####################################################################################
# training related
optim_group = arg_parser.add_argument_group('optimization', 'optimization setting') 
optim_group.add_argument('--decay_rate', default=0.1, type=float, metavar='N', help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--optimizer', default='sgd',  choices=['sgd', 'rmsprop', 'adam'], metavar='N', help='optimizer (default=sgd)')


optim_group.add_argument('--img_load', default='pil_loader' ,  metavar='N',  help='pil_loader or normalizeStaining')
optim_group.add_argument('--pretrained', default= True,   metavar='N', help='pretrained model')
optim_group.add_argument('--train_from_seg_pt', default= False,   metavar='N', help='pretrained model')

optim_group.add_argument('--gpu_number', default= '1,2,3,5', metavar='N', help='gpu select, et. 0,1,2,3 or 4,5,6,7') 
optim_group.add_argument('--gpus', default=4, type=int, metavar='N',  help='gpu number (default: 4)') 
optim_group.add_argument('--nproc_per_node', default=4, type=int, metavar='N',  help='node rank for distributed training') 

optim_group.add_argument('--local_rank', default=0, type=int,  help='node rank for distributed training') 
optim_group.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 512-256,128,256-512)')
optim_group.add_argument('-w', '--num_workers', default=2, type=int,  metavar='N', help='number of data loading workers (default: 4)')
optim_group.add_argument('--cls', default= False,   metavar='N', help='classifier')
optim_group.add_argument('--seg', default= False,   metavar='N', help='segmentation')
optim_group.add_argument('--res', default= False,   metavar='N', help='regress')
optim_group.add_argument('--lubda', default=1, type=float, metavar='N',  help='lubda of the second loss (default: 1)') 
optim_group.add_argument('--epochs', default=60, type=int, metavar='N',  help='number of total epochs to run (default: 164)') 
optim_group.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr_gt', '--gt-learning-rate', default=0, type=float, metavar='LR_gt', help='initial learning rate (default: 0.1)')
optim_group.add_argument('--cos_T_max', default=60, type=int, metavar='N', help='cos lenrning rate T_max ')

optim_group.add_argument('--resize', default= False,   metavar='N', help='resize input')
optim_group.add_argument('-r', '--resize_size', default=512, type=int, metavar='N', help='resize input size')
optim_group.add_argument('--distribute', default= False,   metavar='N', help='distribute training')
optim_group.add_argument('--stainnorm', default= False,   metavar='N', help='distribute training')
optim_group.add_argument('--transform_input', default= False,   metavar='N', help='False:mean0.5, True:mean0.2')
optim_group.add_argument('--model_name', default= 'res50_cls', metavar='N', help='model name select') 

# 数据路径
dataroot = '/../../../camelyon16/figure_512_tile_random_stainnorm/dataframe_stainnorm_addroot/'  
logroot = '/../../../pytorch/gap/train_with_fpn_distribute/logs/camelyon512/'
optim_group.add_argument('--img_root', default= '/../../camelyon16/figure_512_tile_random_stainnorm',  help='img dir root') 
optim_group.add_argument('--log_dir', default= logroot + 'with_0_5/camelyon512_alldata_res50_fpn_stainnorm_w0_5_lr0.01_T60_mean0.5/',  help='tensorboard log dir')
optim_group.add_argument('--train_data_dir', default= dataroot + 'train/train_with_0_5.csv' ,  metavar='N', help='train dataframe dir')
optim_group.add_argument('--val_data_dir', default= dataroot +'val/val_with_0_5_sample.csv',   metavar='N', help='val dataframe dir')
optim_group.add_argument('--test_data_dir', default= dataroot + 'val/val_with_0_5.csv' ,   metavar='N', help='test dataframe dir') 

 ################
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum (default=0.9)')
optim_group.add_argument('--no_nesterov', dest='nesterov', action='store_false',  help='do not use Nesterov momentum')
optim_group.add_argument('--alpha', default=0.99, type=float, metavar='M',  help='alpha for ')
optim_group.add_argument('--beta1', default=0.9, type=float, metavar='M',  help='beta1 for Adam (default: 0.9)')
optim_group.add_argument('--beta2', default=0.999, type=float, metavar='M',  help='beta2 for Adam (default: 0.999)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

###################################################################



exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--resume', default=False,   metavar='N', help='resume, continue latest checkpoint training')
exp_group.add_argument('--resume_file', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)') 
exp_group.add_argument('--pt_path', default='', type=str, metavar='PATH', help='path to pretrained checkpoint (default: none)') 
exp_group.add_argument('-f', '--force', dest='force', action='store_true', help='force to overwrite existing save path')
exp_group.add_argument('--seed', default=0, type=int, help='random seed') 

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')   
# data_group.add_argument('-j', '--workers', dest='num_workers', default=2,
#                         type=int, metavar='N',
#                         help='number of data loading workers (default: 4)')    

# model arch related
arch_group = arg_parser.add_argument_group('arch',  'model architecture setting') 
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='',  type=str, choices='resnet50',  help='model architecture: ' + ' | '.join('resnet50') + ' (default: resnet)')
arch_group.add_argument('--drop-rate', default=0.0, type=float,  metavar='DROPRATE', help='dropout rate (default: 0.2)')
arch_group.add_argument('--death-mode', default='none', choices=['none', 'linear', 'uniform'],  help='death mode for stochastic depth (default: none)')
arch_group.add_argument('--death-rate', default=0.5, type=float,  help='death rate rate (default: 0.5)') 

# used to set the argument when to resume automatically
arch_resume_names = ['arch', 'depth', 'death_mode', 'death_rate', 'death_rate',
                     'growth_rate', 'bn_size', 'compression']

