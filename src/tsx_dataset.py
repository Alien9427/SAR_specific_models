import read_dataset
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import time


class TSX_Dataset(Dataset):
    """
    TerraSAR-X dataset
    """

    def __init__(self, txt_file, root_dir, data_type, transform = None):
        """
        Args:
            :param txt_file: path to the txt file with ($path$ $label$)
            :param root_dir: full_path = root_dir + file_path
            :param transform: optional transform to be applied on a sample
        """
        self.tsx_path_label = read_dataset.read_dataset_txt(txt_file)
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform

    def __len__(self):
        return len(self.tsx_path_label)

    def __getitem__(self, idx):
        """
        Args:
            :param idx: the index of data
            :param data_type: "npy" or "tif"
            :return:
        """

        if self.data_type[0:3] == 'npy':
            patch_path = os.path.join(self.root_dir, 'TSXdataset_' + self.data_type, self.tsx_path_label[idx][0]) + '.npy'
            image = read_dataset.read_npy(patch_path)
        elif self.data_type == 'tif':
            patch_path = os.path.join(self.tsx_path_label[idx][0]) + '.jpg'
            image = read_dataset.read_tif(self.root_dir + '/TSXdatasetOctavian', patch_path)
        elif self.data_type == 'jpg':
            patch_path = os.path.join(self.root_dir, 'TSXdatasetOctavian', self.tsx_path_label[idx][0]) + '.jpg'
            image = read_dataset.read_jpg(patch_path)

        label = self.tsx_path_label[idx][1]
        sample = {'image' : image, 'label' : label, 'path': self.tsx_path_label[idx][0]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class Target_Dataset(Dataset):
    """
    for MSTAR and OpenSARShip dataset
    """

    def __init__(self, txt_file, transform=None):
        self.path_label = read_dataset.read_dataset_txt(txt_file)
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):

        image = read_dataset.read_sar_target(self.path_label[idx][0])
        label = self.path_label[idx][1]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label, 'path': self.path_label[idx][0]}