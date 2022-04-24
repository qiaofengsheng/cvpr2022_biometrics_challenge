'''
 ==================板块功能描述====================
           @Time     :2022/4/23   22:39
           @Author   : qiaofengsheng
           @File     :get_feature.py
           @Software :PyCharm
           @description:
 ================================================
 '''
import os

import torch
from PIL import Image
from torch import nn
import numpy as np
import pandas as pd
from torchvision import transforms
import tqdm

data=pd.read_csv(r'H:\pet_biometric_challenge_2022\pet_biometric_challenge_2022\train\train_data.csv')
from model import iresnet

net=iresnet.iresnet50()
net.load_state_dict(torch.load('params/best.pth'))
net.eval()
from utils import *
for i in tqdm.tqdm(data.loc[:,'nose print image']):
    if i.__contains__('*'):
        i=i.replace('*','_')
    img=Image.open(os.path.join(r'H:\pet_biometric_challenge_2022\pet_biometric_challenge_2022\train\images',i))
    img=keep_resize(img)
    trasform = transforms.Compose([
        transforms.ToTensor()])
    img=trasform(img)
    img=torch.unsqueeze(img,dim=0)
    feature=net(img)
    np.save(os.path.join('data/train_feature',i.replace('.jpg','.npy')),feature.detach().numpy())



