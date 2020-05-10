import torch
from torch.utils.data import Dataset,DataLoader

import PIL.Image as Image
import os
from torchvision import datasets, models, transforms
import numpy as np

class MyDataset(Dataset):
    '''
    Attention: length of datay and dataxgt must be the same
    TODO: 增加alert机制防止数据长度不一致
    '''
    def __init__(self,root_dir,transform=None):
        '''
        TODO add None transform alert
        '''
        self.datay=[]
        self.dataxgt=[]
        self.transform=transform
        y_dir = root_dir + '/y'
        xgt_dir = root_dir + '/xgt'
        for root, dirs, files in os.walk(y_dir):
            for file in files:
                if file.split('.')[-1] in ['png','bmp']:
                    img=Image.open(y_dir+'/'+file).convert('L')
                    self.datay.append(self.transform(img))#自动归一化了
        for root, dirs, files in os.walk(xgt_dir):
            for file in files:
                if file.split('.')[-1] in ['png','bmp']:
                    img=Image.open(xgt_dir+'/'+file).convert('L')
                    self.dataxgt.append(self.transform(img))#自动归一化了          

    def __len__(self):
        return len(self.datay)

    def __getitem__(self, item):
        y= self.datay[item]
        xgt = self.dataxgt[item]
        return y, xgt

class CreateKernel(Dataset):
    def __init__(self,kernel_path,num):
        self.data=[]
        img=Image.open(kernel_path)
        for i in range(num):
            self.data.append(transforms.ToTensor()(img))#自动归一化了

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data= self.data[item]
        return data

class CreateKernel_T(Dataset):
    def __init__(self,kernel_path,num):
        self.data=[]
        img=Image.open(kernel_path)
        A = transforms.ToTensor()(img)
        At = self.transpose_kernel(A)
        for i in range(num):
            self.data.append(At)#自动归一化了

    def transpose_kernel(self,k):
        return np.fliplr(np.flipud(k))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data= self.data[item]
        return data

if __name__=='__main__':
    root='/Users/zhanggezhi/Deblur/datasets'
    image_datasets = MyDataset(root)
    dataloaders = DataLoader(image_datasets, batch_size=25, shuffle=True, num_workers=4)
    print(dataloaders)
    for y,xgt in dataloaders:
        print(y,xgt)
    # print(len(dset))
    # kset = CreateKernel_T('../kernel.bmp',10)
    # for k in kset:
    #     print(k)
    # # for e in dset:
    # #     img=transforms.ToPILImage()(e)
    #
    # dataiter=DataLoader(dset,batch_size=4,shuffle=True)
    # for e in dataiter:
    #     print(e)
    # img=Image.open('11.png')
    # w,h=img.size
    # img_=torch.zeros(w,h)
    # noise = np.random.normal(mean, var ** 0.5, (w,h))
    # print(img_)




