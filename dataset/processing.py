'''
 ==================板块功能描述====================
           @Time     :2022/4/23   16:05
           @Author   : qiaofengsheng
           @File     :processing.py
           @Software :PyCharm
           @description:
 ================================================
 '''
import random

import numpy as np
import pandas as pd
import tqdm


def precessing_data(data_csv_path, is_train=True):
    dataset = []
    data = pd.read_csv(data_csv_path)
    dog_id = set(data.loc[:, 'dog ID'].tolist())

    for i in tqdm.tqdm(dog_id):
        temp_dog_id = dog_id
        # 制作正样本
        index_ = np.where(data.loc[:, 'dog ID'] == i)
        index = index_[0].tolist()
        if len(index) == 2:
            if data.loc[index[0], 'nose print image'].__contains__('*'):
                data.loc[index[0], 'nose print image'] = data.loc[index[0], 'nose print image'].replace('*', '_')
            if data.loc[index[1], 'nose print image'].__contains__('*'):
                data.loc[index[1], 'nose print image'] = data.loc[index[1], 'nose print image'].replace('*', '_')
            dataset.append([data.loc[index[0], 'nose print image'], data.loc[index[1], 'nose print image'],1])
        else:
            for j in range(len(index) - 1):
                for k in range(j + 1, len(index)):
                    if data.loc[index[k], 'nose print image'].__contains__('*'):
                        data.loc[index[k], 'nose print image'] = data.loc[index[k], 'nose print image'].replace('*',
                                                                                                                '_')
                    if data.loc[index[j], 'nose print image'].__contains__('*'):
                        data.loc[index[j], 'nose print image'] = data.loc[index[j], 'nose print image'].replace('*',
                                                                                                                '_')
                    dataset.append([data.loc[index[j], 'nose print image'], data.loc[index[k], 'nose print image'], 1])

        # 制作负样本
        index_negative_ = np.where(data.loc[:, 'dog ID'] != i)
        index_negative = index_negative_[0].tolist()
        for m in random.sample(index_negative, 7):
            if data.loc[m, 'nose print image'].__contains__('*'):
                data.loc[m, 'nose print image'] = data.loc[m, 'nose print image'].replace('*', '_')
            dataset.append([data.loc[index[random.randint(0, len(index) - 1)], 'nose print image'],
                            data.loc[m, 'nose print image'], 0])

    df = pd.DataFrame(dataset, columns=['imageA', 'imageB', 'similar'])
    if is_train:
        df.to_csv('../data/train_data.csv', index=False)


if __name__ == '__main__':
    precessing_data(r'H:\pet_biometric_challenge_2022\pet_biometric_challenge_2022\train\train_data.csv', True)
    # d=[1,2,3,4,5,6,520,2]
    # print(random.sample(d,3))
