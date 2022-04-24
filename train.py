'''
 ==================板块功能描述====================
           @Time     :2022/4/23   15:55
           @Author   : qiaofengsheng
           @File     :train.py
           @Software :PyCharm
           @description:
 ================================================
 '''
import numpy as np
import torch.cuda
import tqdm
from torch import nn, optim
from model import iresnet
from utils import *
from dataset import dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

class Train():
    def __init__(self, model_path='params/last.pth', csv_path='data/train_data.csv',
                 image_root_path=r'H:\pet_biometric_challenge_2022\pet_biometric_challenge_2022\train\images'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = iresnet.iresnet50().to(self.device)
        self.net.load_state_dict(torch.load(model_path))
        self.loss_fun = pet_loss('l1')
        self.opt = optim.AdamW(self.net.parameters())
        self.dataset = dataset.PetDataSet(csv_path=csv_path, image_root_path=image_root_path)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.9), len(self.dataset)-int(len(self.dataset)*0.9)])

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=15, shuffle=True, drop_last=False)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=100, shuffle=True, drop_last=False)
    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            self.net.train()
            with tqdm.tqdm(self.train_data_loader, desc='Train') as tbar:
                for i, (image1, image2, label) in enumerate(self.train_data_loader):
                    image1, image2, label = image1.to(self.device), image2.to(self.device), label.to(self.device)
                    out1 = self.net(image1)
                    out2 = self.net(image2)
                    pred_out = cos_similar(out1, out2)
                    pred_out=torch.sigmoid(pred_out)
                    self.opt.zero_grad()
                    train_loss = self.loss_fun(pred_out, label)
                    train_loss.backward()
                    self.opt.step()
                    tbar.set_postfix(Epoch=epoch,loss=train_loss.item())
                    tbar.update()
                    if (i+1)%100==0:
                        torch.save(self.net.state_dict(), 'params2/last.pth')
                        print('save last checkpoint successfully!')
                    # i=1000
                    if (i+1)%1000==0:
                        with torch.no_grad():
                            auc_val=0
                            c,temp_label,pred=0,[],[]
                            self.net.eval()
                            with tqdm.tqdm(self.val_data_loader, desc='Test') as tbar2:
                                for i, (image1_, image2_, label_) in enumerate(self.val_data_loader):
                                    image1_, image2_, label_ = image1_.cuda(), image2_.cuda(), label_.cuda()
                                    out1_ = self.net(image1_)
                                    out2_ = self.net(image2_)
                                    pred_out_ = cos_similar(out1_, out2_)
                                    pred_out_=torch.sigmoid(pred_out_)
                                    if c==0:
                                        temp_label=label_.cpu().numpy().tolist()
                                        pred=pred_out_.cpu().numpy().tolist()
                                    else:
                                        temp_label += label_.cpu().numpy().tolist()
                                        pred += pred_out_.cpu().numpy().tolist()
                                    c+=1
                                    tbar2.update()
                                auc_val=roc_auc_score(np.array(temp_label),np.array(pred))
                            if auc_val>auc:
                                self.net.train()
                                torch.save(self.net.state_dict(),'params2/best.pth')
                                print('save best checkpoint successfully!')
                                auc=auc_val
                                print(auc)






if __name__ == '__main__':
    Train().train(5)
