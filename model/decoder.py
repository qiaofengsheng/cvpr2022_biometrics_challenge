'''
 ==================板块功能描述====================
           @Time     :2022/4/23   22:30
           @Author   : qiaofengsheng
           @File     :decoder.py
           @Software :PyCharm
           @description:
 ================================================
 '''
from torch import nn


class OutDecoder(nn.Module):
    def __init__(self):
        super(OutDecoder, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.feature(x)
