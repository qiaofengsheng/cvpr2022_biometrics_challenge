'''
 ==================板块功能描述====================
           @Time     :2022/4/23   16:00
           @Author   : qiaofengsheng
           @File     :dataset.py
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


class PetDataSet(Dataset):
    def __init__(self, csv_path, image_root_path):
        self.image_root_path = image_root_path
        self.data = np.array(pd.read_csv(csv_path).loc[:, ['imageA', 'imageB', 'similar']])
        self.trasform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image1 = os.path.join(self.image_root_path, self.data[index, 0])
        image2 = os.path.join(self.image_root_path, self.data[index, 1])
        img1 = keep_resize(Image.open(image1))
        img2 = keep_resize(Image.open(image2))
        return self.trasform(img1), self.trasform(img2), self.data[index, 2]


class FeatureDataSet(Dataset):
    def __init__(self, csv_path,root_path):
        self.root_path=root_path
        self.data = np.array(pd.read_csv(csv_path).loc[:, ['imageA', 'imageB', 'similar']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature1 = self.data[index, 0]
        feature2 = self.data[index, 1]
        return torch.Tensor(np.load(os.path.join(self.root_path,feature1))).squeeze(dim=0), torch.Tensor(np.load(os.path.join(self.root_path,feature2))).squeeze(dim=0), self.data[index, 2]


if __name__ == '__main__':
    # data = PetDataSet(r'../data/train_data.csv',
    #                   r'H:\pet_biometric_challenge_2022\pet_biometric_challenge_2022\train\images')
    # dataloader = DataLoader(data, batch_size=1000)
    #
    # for image1, image2, label in tqdm.tqdm(dataloader):
    #     print(image1.shape, image2.shape, label)

    d=FeatureDataSet(r'../data/train_feature_data.csv','../data/train_feature')
    dataloader = DataLoader(d, batch_size=2,shuffle=False)
    for i1,i2,l in tqdm.tqdm(dataloader):
        print(i1.shape,i2.shape,l.shape)

