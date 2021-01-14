import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0 
        self.sum = 0
        self.count = 0
        # self.val = torch.tensor(0).float()
        # self.avg = torch.tensor(0).float()
        # self.sum = torch.tensor(0).float()
        # self.count = torch.tensor(0).float()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class Evaluator(object):
    def __init__(self, num_class, cuda=True):
        self.num_class = num_class
        self.cuda=cuda
        if self.cuda:
            self.confusion_matrix = torch.zeros((self.num_class,) * 2).cuda()
        else:
            self.confusion_matrix = torch.zeros((self.num_class,) * 2) 

    def Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc*100


    def Pixel_Accuracy_Class(self):
        Acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # Acc = torch.nanmean(Acc) # 需要改
        Acc = torch.mean(Acc) # 需要改
        return Acc*100
 
    def Mean_Intersection_over_Union(self):
        MIoU = torch.diag(self.confusion_matrix) / (
                    torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                    torch.diag(self.confusion_matrix))
        MIoU = torch.mean(MIoU)
        # MIoU = torch.nanmean(MIoU)
        return MIoU


    def Frequency_Weighted_Intersection_over_Union(self):
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = torch.diag(self.confusion_matrix) / (
                    torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                    torch.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


    def _generate_matrix(self, gt_image, pre_image): 
        mask = (gt_image >= 0) & (gt_image < self.num_class)  
        label = self.num_class * gt_image[mask].int() + pre_image[mask] 

        count = torch.bincount(label, minlength=self.num_class**2) 
        confusion_matrix = count.reshape(self.num_class, self.num_class) 
        return confusion_matrix


    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        for lp, lt in zip(pre_image, gt_image):             
            self.confusion_matrix += self._generate_matrix(lt.flatten(), lp.flatten())


    def reset(self):
        if self.cuda:
            self.confusion_matrix = torch.zeros((self.num_class,) * 2).cuda()
        else:
            self.confusion_matrix = torch.zeros((self.num_class,) * 2) 
 
if __name__=='__main__':
    
    import torch     
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    output = torch.tensor([[[[0.12,0.36],
                                [0.22,0.66]],
                                [[0.13,0.34],
                                [0.52,-0.96]]],
                            [[[0.12,0.36],
                                [0.22,0.66]],
                                [[0.13,0.34],
                                [0.52,-0.96]]],
                            [[[0.12,0.36],
                                [0.22,0.66]],
                                [[0.13,0.34],
                                [0.52,-0.96]]]])

    target0 = torch.tensor([[[1,0],
                            [0,1]],
                        [[0,1],
                            [0,1]],
                        [[1,0],
                            [1,1]]])  
    output = output.cuda()
    target = target0.cuda() 
    t1=time.time()

    for i in range(100):
        pred = torch.argmax(output, axis=1)  
        evaluator=Evaluator(num_class=2)
        evaluator.add_batch(target, pred)  
    print(time.time()-t1)
    print(evaluator.Mean_Intersection_over_Union())
    print(evaluator.Pixel_Accuracy())
      