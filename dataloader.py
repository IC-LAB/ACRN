"""
pytorch加载数据
MyDataset(loader='pil_loader') ==> directly load img
MyDataset(loader='normalizeStaining')  ==> load img & stain normalization,refer to https://github.com/schaugf/HEnorm_python
# 加入了随机crop和旋转，同时旋转mask和image(需要注意修改init函数初始值，图像块大小，默认是512crop成448)
return img，mask, label,tumor_p 
""" 

import pandas as pd
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image 
import numpy as np
import torchvision.transforms.functional as F
import random

# import matplotlib.pyplot as plt
import os 
import torch 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms  

 
#  data load
def pil_loader(path):    # 一般采用pil_loader函数。
    with open(path, 'rb') as f:
        with Image.open(f) as img: 
            return img.convert('RGB')
def pil_loader_mask(path):    # 一般采用pil_loader函数。 
    with open(path, 'rb') as f:
        with Image.open(f) as img: 
            return img.convert('L')   

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path) 


def Random_Vertical_Flip(img,mask,p=0.5): #,mask
    if mask:
        if random.random() < p: 
                return F.vflip(img),F.vflip(mask),
        return img,mask
    else:
        if random.random() < p: 
                return F.vflip(img)
        return img

def Random_Horizontal_Flip(img,mask,p=0.5): #,mask
    if mask:
        if random.random() < p:
            return F.hflip(img),F.hflip(mask),
        return img,mask 
    else:
        if random.random() < p:
            return F.hflip(img)
        return img

    
class MyDataset(Dataset):
    def __init__(self, dataframe, dataset='', loader='pil_loader', data_transforms=None, args=None):
        self.img_wh = 448
        self.crop_node = 64 #512-448
        self.resize = args.resize
        self.resize_size = args.resize_size
        self.args =args
        
        # dataframe.sample(frac=1).reset_index(drop=True) # 打乱顺序

        self.img_name = dataframe['img_path'] 
        self.img_name = np.array(self.img_name)#np.ndarray()
        self.img_name = self.img_name.tolist()#list 
        
        if args.seg:
            self.mask_name = dataframe['label_path']
            self.mask_name = np.array(self.mask_name)#np.ndarray()
            self.mask_name = self.mask_name.tolist()#list
        if args.cls:
            self.label = dataframe['label']
            self.label = np.array(self.label)#np.ndarray()
            self.label = self.label.tolist()#list 
        if args.res:
            self.tumor_probability = dataframe['tumor_probability']
            self.tumor_probability = np.array(self.tumor_probability)#np.ndarray()
            self.tumor_probability = self.tumor_probability.tolist()

        self.data_transforms = data_transforms
        self.dataset=dataset

        if loader =='normalizeStaining':
            self.loader = normalizeStaining
            print("normalizeStaining")
        elif loader== 'pil_loader':
            self.loader = pil_loader
        self.loader_mask = pil_loader_mask

    def __getitem__(self, index):
        # 这里可以对数据进行处理,比如讲字符数值化 
        img_name = self.args.img_root+self.img_name[index]   
        img = self.loader(img_name)    
        if self.resize:
            img = F.resize(img, self.resize_size) 
        
        if self.args.cls:
            label = self.label[index]  
        else:
            label = 1
        if self.args.res:
            tumor_probability = self.tumor_probability[index]
            prob = np.array([tumor_probability]) 
        else:
            prob = np.array([1.0])
        if self.args.seg:
            mask_name = self.args.img_root + self.mask_name[index] 
            mask = self.loader_mask(mask_name)   
            if self.resize: 
                mask = F.resize(mask, self.resize_size) 
        
        # img,mask = rand_crop(img,mask,img_wh=self.img_wh,train_or_test=self.dataset,crop_node=self.crop_node)
        if self.dataset=='train': 
            if self.args.seg:
                img, mask = Random_Vertical_Flip(img,mask,p=0.5)
                img, mask = Random_Horizontal_Flip(img,mask,p=0.5) 
            else: 
                img = Random_Vertical_Flip(img,None,p=0.5)
                img = Random_Horizontal_Flip(img,None,p=0.5) 
        if not self.args.seg:
            mask = [1]

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:  
                print("Cannot transform image: {}".format(img_name))   
        
        return img,np.array(mask),int(label), prob.astype(np.float32) 

    def __len__(self):
        return len(self.img_name) 



def normalizeStaining(img_path, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    # define height and width of image
    # img = Image.open(img_path)
    # plt.imshow(img)
    # plt.show()
    img = np.array(Image.open(img_path))
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    return Image.fromarray(Inorm.astype('uint8')).convert('RGB')#Inorm
 
###################################################
# '''test code'''
# if __name__ == "__main__": 
#     from args import arg_parser, arch_resume_names
#     global args 
#     args = arg_parser.parse_args()  
#     args.seg = True
#     args.cls = True
#     args.res = True
     
#     transform = {'train':transforms.Compose([transforms.ToTensor(),
#                                              transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) ]), #(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#                'test':transforms.Compose([transforms.ToTensor(),
#                                           transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))  ]) # (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#                }  

#     train_data_dir = "/mnt1/trx/pytorch/gap/train_with_distribute/test.csv" 
#     args.img_root = '../../../camelyon16/figure_512_tile_random_stainnorm/'
#     train_records = pd.read_csv(train_data_dir,header=0)

#     print(len(train_records)) 

#     batch_size = 1
#     trainset = MyDataset( dataframe=train_records,dataset='train', data_transforms = transform )
#     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    
#     for j in range(4):
#         for i, data in enumerate(trainloader): 
#             print(data[-2],data[-1])


