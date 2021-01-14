''' 训练配置函数
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time
import sys 
import time
import os
import shutil 
import torch
import numpy as np
from colorama import Fore
import torch.nn.functional as F
from Evaluator_gpu import *  
from dataloader import *

def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                beta=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch >= int(num_epochs * 0.66):
        lr *= decay_rate**2
    elif epoch >= int(num_epochs * 0.33):
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

    #     res = []
    #     for k in topk:
    #         correct_k = correct[:k].view(-1).float().sum(0)
    #         res.append(100.0 - correct_k.mul_(100.0 / batch_size)) 
    # return res  # res[topk[0],topk[1]]
        k=topk[0]
        # print('k',k)
        correct_k = correct[:k].view(-1).float().sum(0)
        res = 100.0 - correct_k.mul_(100.0 / batch_size)
    return res 


class Trainer(object):
    def __init__(self, model, criterion_cls=None, criterion_res=None,criterion_seg=None, optimizer=None, args=None):
        self.model = model
        self.criterion_cls = criterion_cls
        self.criterion_seg = criterion_seg
        self.criterion_res = criterion_res

        self.optimizer = optimizer 
        self.args = args 
        self.evaluator_cls = Evaluator(num_class=2)
        self.evaluator_seg = Evaluator(num_class=2) 
        self.evaluator_seg_cls = Evaluator(num_class=2)
        
    def train(self, train_loader, epoch): 
        losses_total = AverageMeter()
        losses_cls = AverageMeter()
        losses_seg = AverageMeter()
        losses_res = AverageMeter()
        # top1 = AverageMeter()
        
        self.evaluator_cls.reset()
        self.evaluator_seg.reset()
        self.evaluator_seg_cls.reset()
      
        # switch to train mode
        self.model.train()

        lr = adjust_learning_rate(self.optimizer, self.args.lr, self.args.decay_rate, epoch, self.args.epochs)  # TODO: add custom
        # for param_group in self.optimizer.param_groups:
        #     lr = param_group['lr']   #使用cos余弦退火法,已经自动调整lr了，这里返回lr打印                        
         
        for i, (imgs, masks, labels, probs) in enumerate(train_loader): #probs 
            imgs = imgs.cuda( non_blocking=True)
            labels = labels.cuda( non_blocking=True)
            masks = masks.cuda( non_blocking=True) 
            probs = probs.cuda( non_blocking=True)
            if not self.args.cls and not self.args.res and self.args.seg: # seg
                x_seg = self.model(imgs) #x_class,x_res
                loss_seg = self.criterion_seg(x_seg, masks.long() )
                loss_total = loss_seg 
            elif self.args.cls and not self.args.res and self.args.seg: # cls + seg
                x_cls, x_seg = self.model(imgs) #x_class,x_res
                loss_cls = self.criterion_cls(x_cls, labels ) 
                loss_seg = self.criterion_seg(x_seg, masks.long() )
                loss_total = loss_cls + self.args.lubda * loss_seg #loss_res
            elif self.args.cls and self.args.res and not self.args.seg: # cls + res
                x_cls, x_res = self.model(imgs) #x_class,x_res
                loss_cls = self.criterion_cls(x_cls, labels ) 
                loss_res = self.criterion_res(torch.sigmoid(x_res), probs)
                # loss_res = self.criterion_res(x_res, probs)
                loss_total = loss_cls + self.args.lubda * loss_res #loss_res
            elif self.args.cls and not self.args.res and not self.args.seg: # cls
                x_cls = self.model(imgs) #x_class,x_res
                loss_cls = self.criterion_cls(x_cls, labels) 
                loss_total = loss_cls 
                # print(loss_cls, x_cls, labels)
 
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            #=================计算指标======================  
            if self.args.cls:
                losses_cls.update(loss_cls, imgs.size(0))
            if self.args.seg:
                losses_seg.update(loss_seg, imgs.size(0))
            if self.args.res:
                losses_res.update(loss_res, imgs.size(0))
            losses_total.update(loss_total, imgs.size(0))
            
            # # self.evaluator_seg.add_batch(masks, torch.argmax(x_seg, axis=1))
            # # self.evaluator_seg_cls.add_batch(labels, (torch.mean(torch.mean(torch.argmax(x_seg,dim=1).float(),dim=2),dim=1)>=0.5).int())

        # print('losses_total.avg', losses_total.avg)
        return losses_total.avg, losses_cls.avg, losses_res.avg, losses_seg.avg, lr ,self.optimizer

    def test(self, val_loader, epoch, silence=False): 
        losses_cls = AverageMeter()
        losses_res = AverageMeter()
        losses_seg = AverageMeter()
        losses_total = AverageMeter() 

        # top1 = AverageMeter() 
        self.evaluator_cls.reset()
        self.evaluator_seg.reset()
        self.evaluator_seg_cls.reset()

        # switch to evaluate mode
        self.model.eval()
        
        # initial
        cls_acc, seg_cls_acc, seg_acc, seg_mIoU = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0)
        with torch.no_grad():
            for i, (imgs, masks, labels, probs) in enumerate(val_loader):#probs
                imgs = imgs.cuda( non_blocking=True)
                labels = labels.cuda( non_blocking=True)
                masks = masks.cuda( non_blocking=True)
                probs = probs.cuda() # probs.unsqueeze(1)
         
                # print(imgs.size(),  probs.size(), labels.size()) # torch.Size([bs, 3, 256, 256]) torch.Size([bs, 1]) torch.Size([bs])

                if not self.args.cls and not self.args.res and self.args.seg: # seg 
                    x_seg = self.model(imgs) #x_class,x_regress
                    loss_seg = self.criterion_seg(x_seg, masks.long() )
                    loss_total = loss_seg 
                elif self.args.cls and not self.args.res and self.args.seg: # cls+seg
                    x_cls, x_seg = self.model(imgs) #x_class,x_regress
                    loss_cls = self.criterion_cls(x_cls, labels ) 
                    loss_seg = self.criterion_seg(x_seg, masks.long() )
                    loss_total = loss_cls + self.args.lubda * loss_seg #loss_res
                elif self.args.cls and self.args.res and not self.args.seg: # cls + res
                    x_cls, x_res = self.model(imgs) #x_class,x_regress
                    loss_cls = self.criterion_cls(x_cls, labels ) 
                    loss_res = self.criterion_res(torch.sigmoid(x_res), probs)
                    # loss_res = self.criterion_res(x_res, probs)
                    loss_total = loss_cls + self.args.lubda * loss_res #loss_res
                elif self.args.cls and not self.args.res and not self.args.seg: # cls
                    x_cls = self.model(imgs) #x_class,x_regress
                    loss_cls = self.criterion_cls(x_cls, labels ) 
                    loss_total = loss_cls 
                     
                #===== measure error and record loss===== 
                # err1 = error(x_class.data, labels, topk=(1, )) # 分类错误，可以看看是否和类计算的一样
                # top1.update(err1.item(), imgs.size(0))
                
                if self.args.seg:
                    self.evaluator_seg.add_batch(masks, torch.argmax(x_seg, axis=1))
                    self.evaluator_seg_cls.add_batch(labels, (torch.mean(torch.mean(torch.argmax(x_seg,dim=1).float(),dim=2),dim=1)>=0.5).int())
                    losses_seg.update(loss_seg, imgs.size(0))
                if self.args.cls:
                    self.evaluator_cls.add_batch(labels, torch.argmax(x_cls, axis=1))
                    losses_cls.update(loss_cls, imgs.size(0))
                if self.args.res:
                    losses_res.update(loss_res, imgs.size(0))
                losses_total.update(loss_total, imgs.size(0))
            if self.args.cls:
                cls_acc = self.evaluator_cls.Pixel_Accuracy() # 分类错误
            if self.args.seg:
                seg_cls_acc = self.evaluator_seg_cls.Pixel_Accuracy() # 分割的分类错误
                seg_acc = self.evaluator_seg.Pixel_Accuracy()     #分割的像素精度
                seg_mIoU = self.evaluator_seg.Mean_Intersection_over_Union() 
 
        return losses_total.avg, losses_cls.avg, losses_res.avg, losses_seg.avg, \
                cls_acc, seg_cls_acc, seg_acc, seg_mIoU 

    def inference(self, test_df, val_loader,  transform, df_save_dir):  
        # 测试并保存到dataframe中
        # cls，res的结果保存到df中
        # seg的结果保存某张图像

        losses_cls = AverageMeter()
        losses_res = AverageMeter()
        losses_seg = AverageMeter()
        losses_total = AverageMeter() 
        acc_cls_class = AverageMeter() 
        acc_cls_res_regress = AverageMeter() 
        acc_cls_res_class = AverageMeter()
        # top1 = AverageMeter() 

        self.evaluator_cls.reset()
        self.evaluator_seg.reset()
        self.evaluator_seg_cls.reset()

        # switch to evaluate mode
        self.model.eval()  

        with torch.no_grad(): 
            # for i in range(0,np.shape(test_df)[0]):  
            for i, (imgs, masks, labels, probs) in enumerate(val_loader):#probs
                imgs = imgs.cuda( non_blocking=True)
                labels = labels.cuda( non_blocking=True)
                masks = masks.cuda( non_blocking=True)
                probs = probs.cuda() # probs.unsqueeze(1)
  
                # # # ==========================数据处理=========================
                # # img_path = self.args.img_root + test_df.loc[i,'img_path']
                # # imgs = pil_loader(img_path) 
                # # imgs = F.resize(imgs, self.args.resize_size)
                # # imgs = transform['test'](imgs).unsqueeze(0)
                # # imgs = imgs.cuda()
        
                # # if self.args.cls:
                # #     labels = torch.tensor([int(test_df.loc[i,'label'])])
                # #     labels = labels.cuda()
                # # if self.args.res:
                # #     probs = torch.tensor([test_df.loc[i,'tumor_probability'].astype(np.float32)]).unsqueeze(0)
                # #     probs = probs.cuda() # probs.unsqueeze(1)
                # # if self.args.seg:
                # #     mask_name = self.args.img_root +  test_df.loc[i, 'label_path']
                # #     masks = pil_loader_mask(mask_name)
                # #     masks = torch.tensor(np.array(F.resize(mask, self.args.resize_size))).unsqueeze(0)
                # #     masks = masks.cuda()
                # # # print(imgs.size(), probs.size(), labels.size())
                
                #==========================分类网络==================================
                if self.args.cls and not self.args.res and not self.args.seg: # cls 
                    x_cls = self.model(imgs)
                    # loss_class = self.criterion_class(x_cls, labels) 
                    # losses_class.update(loss_class.item(), imgs.size(0)) 

                    self.evaluator_cls.add_batch(labels, torch.argmax(x_cls, axis=1))     
                    err1_cls_class = error(x_cls.data, labels, topk=(1, ))
                    acc_cls_class.update(100 - err1_cls_class.item(), imgs.size(0))  
                    
                    class_1_p_from_class_model = torch.nn.functional.softmax(x_cls,dim=1)[:,1] #1 类的概率
                    class_0_p_from_class_model = torch.nn.functional.softmax(x_cls,dim=1)[:,0] #0 类的概率
                    
                    for j in range(i*self.args.batch_size,i*self.args.batch_size + imgs.size(0)):
                        test_df.loc[j,'model_cls_class_0'] = class_0_p_from_class_model.cpu()[j - i*self.args.batch_size].item()
                        test_df.loc[j,'model_cls_class_1'] = class_1_p_from_class_model.cpu()[j - i*self.args.batch_size].item()
                    # test_df.loc[i,'model_cls_class_0'] = class_0_p_from_class_model.item()
                    # test_df.loc[i,'model_cls_class_1'] = class_1_p_from_class_model.item()

                #==============================分类+回归网络==================================
                elif self.args.cls and self.args.res and not self.args.seg: # cls+seg
                    x_cls, x_res  = self.model(imgs) 

                    # =================# model_regress  分类分支==========================
                    # loss_cls = self.criterion_cls(x_cls, labels ) 
                    self.evaluator_cls.add_batch(labels, torch.argmax(x_cls, axis=1))      
                    err1_cls_res_class = error(x_cls.data, labels, topk=(1, )) 
                    acc_cls_res_class.update(100 - err1_cls_res_class.item(), imgs.size(0))   

                    #===================== # model_regress 回归分支
                    loss_res = self.criterion_res(torch.sigmoid(x_res), probs) # # loss_res = self.criterion_res(x_res, probs)
                    # loss_total = loss_cls + self.args.lubda * loss_res #loss_res
                    losses_res.update(loss_res, imgs.size(0))

                    acc_regress = torch.sum((torch.sigmoid(x_res).squeeze(1)>0.5).int()==labels)/labels.size(0) 
                    acc_cls_res_regress.update(acc_regress.item()*100, imgs.size(0))   
                    
                    # 保存预测结果到dataframe
                    regress_1_p_from_regress_model =  torch.sigmoid(x_res).squeeze(1) # 回归的概率
                    class_1_p_from_regress_model = torch.nn.functional.softmax(x_cls,dim=1)[:,1] #1 类的概率
                    class_0_p_from_regress_model = torch.nn.functional.softmax(x_cls,dim=1)[:,0] #0 类的概率
                    
                    # print(class_1_p_from_regress_model,i*self.args.batch_size,i*self.args.batch_size + imgs.size(0))
                    for j in range(i*self.args.batch_size, i*self.args.batch_size + imgs.size(0)):
                        test_df.loc[j,'model_cls_res_regress_1'] = regress_1_p_from_regress_model.cpu()[j - i*self.args.batch_size].item()
                        test_df.loc[j,'model_cls_res_class_0'] = class_0_p_from_regress_model.cpu()[j - i*self.args.batch_size].item()
                        test_df.loc[j,'model_cls_res_class_1'] = class_1_p_from_regress_model.cpu()[j - i*self.args.batch_size].item() 

                    # test_df.loc[i,'model_cls_res_regress_1'] = regress_1_p_from_regress_model.item()
                    # test_df.loc[i,'model_cls_res_class_0'] = class_0_p_from_regress_model.item()
                    # test_df.loc[i,'model_cls_res_class_1'] = class_1_p_from_regress_model.item()
                # # ===========================分割网络=======================
                # elif not self.args.cls and not self.args.res and self.args.seg: # seg 
                #     x_seg = self.model(imgs) #x_class,x_regress
                #     loss_seg = self.criterion_seg(x_seg, masks.long() )
                #     loss_total = loss_seg 
                # #============================分割+分类网络===================
                # elif self.args.cls and not self.args.res and self.args.seg: # cls+seg
                #     x_cls, x_seg = self.model(imgs) #x_class,x_regress
                #     loss_cls = self.criterion_cls(x_cls, labels ) 
                #     loss_seg = self.criterion_seg(x_seg, masks.long() )
                #     loss_total = loss_cls + self.args.lubda * loss _seg #loss_res
                else:
                    print('cls res seg error--')
        test_df.to_csv(df_save_dir, index=False, header=True ) 
    
        if self.args.cls and not self.args.res and not self.args.seg: # cls 
            print('Test: model_cls number{num}, cls_acc {cls_acc}, cls_acc_ {cls_acc_}'
                    .format(num = acc_cls_class.count, cls_acc = acc_cls_class.avg, cls_acc_ = self.evaluator_cls.Pixel_Accuracy() ))
        elif self.args.cls and self.args.res and not self.args.seg: # cls + res 
            print('Test: model_cls_res number{num}, cls_acc {cls_acc}, cls_acc_ {cls_acc_}, res_acc {res_acc}'
                    .format(num = acc_cls_res_class.count, cls_acc = acc_cls_res_class.avg, cls_acc_ = self.evaluator_cls.Pixel_Accuracy(),
                            res_acc = acc_cls_res_regress.avg ))

        # elif not self.args.cls and not self.args.res and self.args.seg: # seg  
        # elif self.args.cls and not self.args.res and self.args.seg: # cls+seg 
