'''
 ==================板块功能描述====================
           @Time     :2022/4/23   18:49
           @Author   : qiaofengsheng
           @File     :envaluation.py
           @Software :PyCharm
           @description:
 ================================================
 '''
import os.path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
from utils import *
from model import iresnet
from utils import *


class ValPetDataSet(Dataset):
    def __init__(self, csv_path, image_root_path):
        self.image_root_path = image_root_path
        self.data = np.array(pd.read_csv(csv_path).loc[:, ['imageA', 'imageB']])
        self.trasform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image1 = os.path.join(self.image_root_path, self.data[index, 0])
        image2 = os.path.join(self.image_root_path, self.data[index, 1])
        img1 = keep_resize(Image.open(image1))
        img2 = keep_resize(Image.open(image2))
        return self.data[index, 0], self.data[index, 1], self.trasform(img1), self.trasform(img2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = iresnet.iresnet50().to('cuda')
net.load_state_dict(torch.load('params/best.pth'))
net.eval()
data_loader = DataLoader(
    ValPetDataSet(csv_path=r'H:\pet_biometric_challenge_2022\pet_biometric_challenge_2022\validation\valid_data.csv',
                  image_root_path=r'H:\pet_biometric_challenge_2022\pet_biometric_challenge_2022\validation\images'),
    batch_size=200, shuffle=False)

count=0
temp = []
with torch.no_grad():
    for i1, i2, image1, image2 in tqdm.tqdm(data_loader):
        image1, image2=image1.cuda(), image2.cuda()
        out1 = net(image1)
        out2 = net(image2)
        sim = cos_similar(out1.cpu(), out2.cpu())

        res=np.array(sim).tolist()

        if count==0:
            temp=res
        else:
            temp+=res
        count+=1
    df=pd.read_csv(r'C:\Users\Administrator\Desktop\valid_data.csv')
    df['prediction']=temp
    df.to_csv('valid_data.csv',index=False)

