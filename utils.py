'''
 ==================板块功能描述====================
           @Time     :2022/4/23   16:34
           @Author   : qiaofengsheng
           @File     :utils.py
           @Software :PyCharm
           @description:
 ================================================
 '''
import numpy as np
from torch import nn
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import torch.nn.functional as F
import torch


def pet_loss(names):
    if names == 'l1':
        return nn.L1Loss()
    elif names == 'smooth_l1':
        return nn.SmoothL1Loss()


def keep_resize(image):
    w, h = image.size
    mask = Image.new('RGB', (max(image.size), max(image.size)))
    if w > h:
        mask.paste(image, (0, (w - h) // 2))
    else:
        mask.paste(image, ((h - w) // 2, 0))
    mask = mask.resize((175, 175))
    return mask


def cos_similar(face1, face2):
    face1_norm = F.normalize(face1, dim=1)
    face2_norm = F.normalize(face2, dim=1)
    cosa = torch.matmul(face1_norm, face2_norm.t())
    cosa = torch.diag(cosa)
    return cosa


