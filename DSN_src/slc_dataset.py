from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def read_txt(txt_file):
    pd_data = pd.read_csv(txt_file)
    catename2label = pd.read_csv('../data/catename2label_cate8.txt')
    pd_data['label'] = None

    for i in range(len(pd_data)):
        catename = pd_data.loc[i]['catename']
        label = list(catename2label.loc[catename2label['catename'] == catename]['label'])[0]
        pd_data.loc[i]['label'] = label

    return pd_data

class SLC_spe_4D(Dataset):
    def __init__(self, txt_file, spe_dir, spe_transform=None):
        self.data = read_txt(txt_file)
        self.spe_dir = spe_dir
        self.spe_transform = spe_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slc_spe = np.load(self.spe_dir + self.data.loc[idx]['path'])
        # slc_spe = np.reshape(slc_spe, [32*32, 32, 32])
        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']

        sample = {'spe': slc_spe,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}
        if self.spe_transform:
            sample['spe'] = self.spe_transform(sample['spe'])

        return sample


class SLC_img(Dataset):
    def __init__(self, txt_file, root_dir, transform = None):
        self.data = read_txt(txt_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slc_data = np.load(self.root_dir + self.data.loc[idx]['path'])
        data = np.log2(np.abs(slc_data) + 1) / 16

        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']
        sample = {'data': data,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}

        if self.transform:
            sample['data'] = self.transform(sample['data'])

        return sample

class SLC_spe_xy(Dataset):
    def __init__(self, txt_file, spe_dir, spe_transform=None):
        self.data = read_txt(txt_file)
        self.spe_dir = spe_dir
        self.spe_transform = spe_transform

    def __len__(self):
        return len(self.data) * 32 * 32

    def __getitem__(self, idx):
        a = idx // (32*32)
        b = idx % (32*32)

        slc_spe = np.load(self.spe_dir + self.data.loc[a]['path'])
        slc_spe = np.reshape(slc_spe, [32*32, 32, 32])
        catename = self.data.loc[a]['catename']
        label = self.data.loc[a]['label']

        sample = {'spe': slc_spe[b],
                  'catename': catename,
                  'label': label}
        if self.spe_transform:
            sample['spe'] = self.spe_transform(sample['spe'])

        return sample


class SLC_img_spe4D(Dataset):
    def __init__(self, txt_file, img_dir, spe_dir, img_transform=None, spe_transform=None):
        self.data = read_txt(txt_file)
        self.img_dir = img_dir
        self.spe_dir = spe_dir
        self.img_transform = img_transform
        self.spe_transform = spe_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slc_img = np.load(self.img_dir + self.data.loc[idx]['path'])
        slc_spe = np.load(self.spe_dir + self.data.loc[idx]['path'])

        slc_img = np.log2(np.abs(slc_img) + 1) / 16

        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']
        sample = {'img': slc_img,
                  'spe': slc_spe,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}

        if self.img_transform:
            sample['img'] = self.img_transform(sample['img'])
        if self.spe_transform:
            sample['spe'] = self.spe_transform(sample['spe'])

        return sample