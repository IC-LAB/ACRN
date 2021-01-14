'''
从dataframe的模型的输出结果计算精度
级联测试
'''
import pandas as pd
import glob
import os
import csv
import numpy as np
import torch
# if args.cls and not args.res:
#     df['model_cls_class_0'] = None  
#     df['model_cls_class_1'] = None 
# if args.res:
#     df['model_cls_res_regress_1'] = None 
#     df['model_cls_res_class_0'] = None 
#     df['model_cls_res_class_1'] = None  
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

 
def test_valacc_from_outputcsv():
    '''
    测试验证集的各个模型的精度
    '''
    def val_test(datadir,mode,csv_path ):
        data1= pd.read_csv(datadir)
        if mode=='cls':
            model='model_cls_class_1'
        elif mode == 'res_cls':
            model = 'model_cls_res_class_1'
        elif mode == 'res_res':
            model = 'model_cls_res_regress_1'
          
        result=[]  
        result.append(datadir.split('/')[-3]) 
        result.append(mode)

        top1_acc_0 = AverageMeter()
        top1_acc_1 = AverageMeter()  
        top1_acc_total =  AverageMeter() 

        for i in range(0,np.shape(data1)[0]):   
     
            model_pre = torch.tensor([np.float(data1.loc[i,model])])
            targets = torch.tensor([np.float(data1.loc[i,'label'])]).int()
 
            acc_total = torch.sum((model_pre>=0.5).int()==targets).float()/targets.size(0) 
            
            top1_acc_total.update(acc_total.item()*100, targets.size(0)) 

            if np.float(data1.loc[i,'tumor_probability'] )<0.5:#负样本
                top1_acc_0.update(acc_total.item()*100, targets.size(0)) 
            else: #正样本
                top1_acc_1.update(acc_total.item()*100, targets.size(0)) 
             
        result.append(top1_acc_0.avg)
        result.append(top1_acc_1.avg)
        result.append(top1_acc_total.avg)
        print('datadir{}, acc0:{}, acc1:{}, acc_total:{}'
            .format(datadir.split('/')[-1], top1_acc_0.avg,top1_acc_1.avg, top1_acc_total.avg))
        
        with open(csv_path,'a+') as f:
            csv_write = csv.writer(f) 
            csv_write.writerow(result) 
  
   
def mode2_2data(data1_dir,data2_dir,mode,csv_path):
    '''
    方式2：级联，两个模型进行加权。 
    '''

    data1 = pd.read_csv(data1_dir,header=0)
    data2 = pd.read_csv(data2_dir,header=0) 

    assert np.shape(data1)[0]==np.shape(data2)[0],'error'
    if mode=='cls_rescls':
        model_cls_res1='model_cls_class_1'
        model_cls_res2='model_cls_res_class_1'
    elif mode=='cls_resres':
        model_cls_res1='model_cls_class_1'
        model_cls_res2='model_cls_res_regress_1' 
    elif mode=='rescls_resres':
        model_cls_res1='model_cls_res_class_1'
        model_cls_res2='model_cls_res_regress_1' 

    with open(csv_path,'a+') as f:
        csv_write = csv.writer(f) 
        csv_write.writerow([data1_dir.split('/')[-3], data2_dir.split('/')[-3], mode, model_cls_res1, model_cls_res2]) 

    print('model1:{}, \n model2:{}, \n mode:{}'.format(data1_dir.split('/')[-3], data2_dir.split('/')[-3],mode))
    t = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    t1= [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
    max_ = 0
    result = []
    result.append(data1_dir.split('/')[-3])
    result.append(data2_dir.split('/')[-3])
    result.append(mode)
    result0 = []
    result0.append(data1_dir.split('/')[-3])
    result0.append(data2_dir.split('/')[-3])
    result0.append('acc0')
    result1 = []
    result1.append(data1_dir.split('/')[-3])
    result1.append(data2_dir.split('/')[-3])
    result1.append('acc1')
    result00 = []
    result00.append(data1_dir.split('/')[-3])
    result00.append(data2_dir.split('/')[-3])
    result00.append('acc00')
    result05 = []
    result05.append(data1_dir.split('/')[-3])
    result05.append(data2_dir.split('/')[-3])
    result05.append('acc05')
    result51 = []
    result51.append(data1_dir.split('/')[-3])
    result51.append(data2_dir.split('/')[-3])
    result51.append('acc51')
    result11 = []
    result11.append(data1_dir.split('/')[-3])
    result11.append(data2_dir.split('/')[-3])
    result11.append('acc11')
    for i in range(len(t)):
        lamba = t[i]
        lamba_a = t1[i]

        top1_acc_0 = AverageMeter()
        top1_acc_1 = AverageMeter()  
        top1_acc_total =  AverageMeter()  
        top1_acc_00 = AverageMeter()
        top1_acc_11 = AverageMeter()  
        top1_acc_51 = AverageMeter()  
        top1_acc_05 = AverageMeter()  

        for i in range(0,np.shape(data1)[0]):   
            assert data1.loc[i,'img_path']==data2.loc[i,'img_path'],'img_path error'

            model1 = torch.tensor([np.float(data1.loc[i,model_cls_res1])])
            model2 = torch.tensor([np.float(data2.loc[i,model_cls_res2])])
            targets = torch.tensor([np.float(data1.loc[i,'label'])]).int()

            total_pred_1 = lamba_a*model1 + lamba*model2
            acc_total = torch.sum((total_pred_1>=0.5).int()==targets).float()/targets.size(0) 
            top1_acc_total.update(acc_total.item()*100, targets.size(0)) 

            # if np.float(data1.loc[i,'tumor_probability'] )<0.5:#负样本
            #     assert data1.loc[i,'label']==0,'label error'
            #     top1_acc_0.update(acc_total.item()*100, targets.size(0)) 
            # else: #正样本
            #     top1_acc_1.update(acc_total.item()*100, targets.size(0)) 

            if np.float(data1.loc[i,'tumor_probability'] )==0:#负样本 
                assert data1.loc[i,'label']==0,'label error'
                top1_acc_00.update(acc_total.item()*100, targets.size(0)) 
                top1_acc_0.update(acc_total.item()*100, targets.size(0)) 
            elif 0<np.float(data1.loc[i,'tumor_probability'] )<0.5: #
                top1_acc_05.update(acc_total.item()*100, targets.size(0)) 
                top1_acc_0.update(acc_total.item()*100, targets.size(0)) 
            elif 0.5<=np.float(data1.loc[i,'tumor_probability'] )<1: 
                top1_acc_51.update(acc_total.item()*100, targets.size(0)) 
                top1_acc_1.update(acc_total.item()*100, targets.size(0)) 
            elif np.float(data1.loc[i,'tumor_probability'] )==1: #正样本
                top1_acc_11.update(acc_total.item()*100, targets.size(0)) 
                top1_acc_1.update(acc_total.item()*100, targets.size(0)) 
            
            
        if top1_acc_total.avg > max_:
            max_ = top1_acc_total.avg
            p_max = lamba

        result.append(top1_acc_total.avg)
        result0.append(top1_acc_0.avg)
        result1.append(top1_acc_1.avg)
        result00.append(top1_acc_00.avg)
        result05.append(top1_acc_05.avg)
        result51.append(top1_acc_51.avg)
        result11.append(top1_acc_11.avg)
        
        print('{}*model1 + {}*model2, acc0:{}, acc1:{}, acc_total:{}'
            .format(lamba_a, lamba, top1_acc_0.avg, top1_acc_1.avg, top1_acc_total.avg))
        print('normal patch{}, tumor patch{}, total patch{}'.format(top1_acc_0.count, top1_acc_1.count, top1_acc_total.count))
        print('00 patch{}, 05 patch{}, 51 patch{}, 11 patch{}'.format(top1_acc_00.count, top1_acc_05.count, top1_acc_51.count, top1_acc_11.count))
    result.append(max_)  
    result.append(p_max)  
    result0.append(max_)  
    result0.append(p_max)  
    result1.append(max_)  
    result1.append(p_max)  
    result00.append(max_)  
    result00.append(p_max)  
    result05.append(max_)  
    result05.append(p_max)  
    result51.append(max_)  
    result51.append(p_max)  
    result11.append(max_)  
    result11.append(p_max)  
    print('max acc total:',max,p_max)
    with open(csv_path,'a+') as f:
        csv_write = csv.writer(f) 
        csv_write.writerow(result) 
        csv_write.writerow(result0)  
        csv_write.writerow(result1) 
        csv_write.writerow(result00)   
        csv_write.writerow(result05)  
        csv_write.writerow(result51)  
        csv_write.writerow(result11)  

def mode2_2data_main():

    def mode2_test(data_cls, data_res, csv_path ):  
         
        data1_dir = data_cls
        data2_dir = data_res
        mode = 'cls_rescls'
        print('model:{},model:{}'.format(data_cls.split('/')[-3],data_res.split('/')[-3]),mode)
        mode2_2data(data1_dir=data1_dir,data2_dir=data2_dir,mode=mode,csv_path=csv_path)  
        
        mode = 'cls_resres'
        print('model:{},model:{}'.format(data_cls.split('/')[-3],data_res.split('/')[-3]),mode)
        mode2_2data(data1_dir=data1_dir,data2_dir=data2_dir,mode=mode,csv_path=csv_path)    

        data1_dir = data_res
        data2_dir = data_res
        mode = 'rescls_resres'
        print('model:{},model:{}'.format(data_res.split('/')[-3],data_res.split('/')[-3]), mode)  
        mode2_2data(data1_dir=data1_dir,data2_dir=data2_dir,mode=mode,csv_path=csv_path)  
          
    roots = [
            # '/mnt/trx/pytorch/gap/train_with_distribute/logs/camelyon512/with_0_5/stainnorm_mean/',
            #  '/mnt/trx/pytorch/gap/train_with_distribute/logs/camelyon512/with_0_5/stainnorm_mean0.5/',
             '/mnt/trx/pytorch/gap/train_with_distribute/logs/paip512/with_0_5/stainnorm_mean/',
            #  '/mnt/trx/pytorch/gap/train_with_distribute/logs/paip512/with_0_5/stainnorm_mean0.5/',
               ]
    for root in roots:# '/mnt/trx/pytorch/gap/train_with_distribute/logs/camelyon512/with_0_5/stainnorm_mean/'
        csv_path = root + 'predict.csv'
        model_paths = glob.glob(root + "cls_res/*/*/predict.csv")
        for model_path in model_paths:
            data_res = model_path 
            data_cls = model_path.replace('cls_res','cls')
            # csv_path='./del.csv'
            mode2_test(data_cls, data_res, csv_path )
            # break
        print('finish', csv_path)
        # break

    # root = '/mnt/trx/pytorch/gap/train_with_distribute/logs/camelyon512/with_0_5/stainnorm_mean0.5/'
    # csv_path = root + 'predict.csv'
    # model_paths = glob.glob(root + "cls_res/*/*/predict.csv")
    # for model_path in model_paths:
    #     data_res = model_path 
    #     data_cls = model_path.replace('cls_res','cls')
    #     mode2_test(data_cls, data_res, csv_path )
    # print(csv_path) 

mode2_2data_main()
 