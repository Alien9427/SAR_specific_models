import numpy as np
import os
from slc_functions import gen_spectrogram_2
import pandas as pd
import random



def gen_all_spec(slc_root, spe_root):
    slc_list = os.listdir(slc_root)
    if not os.path.exists(spe_root):
        os.mkdir(spe_root)
    for cate in slc_list:
        if not os.path.exists(spe_root + cate):
            os.mkdir(spe_root + cate)

        data_list = os.listdir(slc_root + cate)
        for data in data_list:
            if not os.path.exists(spe_root + cate + '/' + data):
                print(data)
                slc_data = np.load(slc_root + cate + '/' + data)
                spectrogram = np.log(1+np.abs(gen_spectrogram_2(slc_data, 32)))

                np.save(spe_root + cate + '/' + data, spectrogram)


def get_range_spec(spe_root):
    max_value = 0
    min_value = 100
    spe_list = os.listdir(spe_root)
    for cate in spe_list:
        spec_list = os.listdir(spe_root + cate)
        for spec in spec_list:
            spectrogram = np.load(spe_root + cate + '/' + spec)
            max_value = max(max_value, spectrogram.max())
            min_value = min(min_value, spectrogram.min())

    print(max_value, min_value)


def get_mean_std(spe_root):
    spe_list = os.listdir(spe_root)
    mean = np.array([0, 0])
    std = np.array([0, 0])
    count = 0
    for cate in spe_list:
        spec_list = os.listdir(spe_root + cate)
        for spec in spec_list:
            spectrogram = np.load(spe_root + cate + '/' + spec)
            mean[0] += spectrogram[:,:,16,:].mean()
            mean[1] += spectrogram[:,:,:,16].mean()
            std[0] += spectrogram[:,:,16,:].std()
            std[1] += spectrogram[:,:,:,16].std()
            count += 1

    print(mean / count, std / count)

def get_mean_std_xy(spe_root):
    spe_list = os.listdir(spe_root)
    mean = 0
    std = 0
    count = 0
    for cate in spe_list:
        spec_list = os.listdir(spe_root + cate)
        for spec in spec_list:
            spectrogram = np.load(spe_root + cate + '/' + spec)
            x, y, fr, fa = spectrogram.shape
            spectrogram = spectrogram.reshape([x*y, fr, fa])
            for i in range(x*y):
                mean += spectrogram[i, :, :].mean()
                std += spectrogram[i, :, :].std()
                count += 1

    print(mean / count, std / count)

def gen_train_val(data_root):
    val_ratio = 0.7

    df_train = pd.DataFrame(columns=['path', 'catename'])
    df_val = pd.DataFrame(columns=['path', 'catename'])

    # val_num = 95
    for cate in os.listdir(data_root):
        data_list = os.listdir(data_root + cate)
        random.shuffle(data_list)
        val_num = int(len(data_list) * val_ratio)
        for i, item in enumerate(data_list):
            if i < val_num:
                df_val.loc[len(df_val) + 1] = [cate + '/' + item, cate]
            else:
                df_train.loc[len(df_train) + 1] = [cate + '/' + item, cate]

    df_train.to_csv('../data/slc_train_22.txt', index=False)
    df_val.to_csv('../data/slc_val_22.txt', index=False)


def gen_hv_data(data_txt):
    df_data = pd.read_csv(data_txt)
    df_data_new = pd.DataFrame(columns=['path', 'catename'])

    total_num = len(df_data)
    for i in range(total_num):
        data_path = df_data.loc[i]['path']
        catename = df_data.loc[i]['path']
        img = data_path[:-4].split('_')[1]
        loc = data_path[:-4].split('_')[3:4]



if __name__ == '__main__':
    slc_root = '../data/slc_data/'
    spe_root = '../data/spe_data/'
    gen_all_spec(slc_root, spe_root)
    # get_range_spec(spe_root)
    # gen_train_val(slc_root)
    get_mean_std_xy(spe_root)
    # gen_hv_data('../data/slc_val_3.txt')