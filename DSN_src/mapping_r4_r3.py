import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import transform_data
from torch import optim, nn
import network
from slc_dataset import SLC_spe_4D
import numpy as np
import os.path
import argparse

"""
mapping_r4_r3.py
Map the 4-D signals to 3-D tensors, with the pre-trained cae model ../model/slc_spexy_cae_3.pth
The 3-D tensors are saved in ../data/spexy_data_3
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='arg_train')

    parser.add_argument('--data_txt', default='../data/slc_train_3.txt')
    args = parser.parse_args()
    txt_file = args.data_txt

    # txt_file = '../data/slc_train_3.txt'
    data_transforms = transforms.Compose([
        transform_data.Normalize_spe_xy(),
        transform_data.Numpy2Tensor()
    ])
    batch_size = 2

    save_path = '../data/spexy_data_3/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    net = network.SLC_spexy_CAE()
    net.load_state_dict(torch.load('../model/slc_spexy_cae_3.pth'))

    dataset = SLC_spe_4D(txt_file=txt_file, spe_dir='../data/spe_data/', spe_transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # spe_3D = torch.Tensor(np.zeros([batch_size, 128, 32, 32]))

    # for sample in dataloader:
    #     data = sample['spe']
    #     for x in range(32):
    #         for y in range(32):
    #             data_xy = data[:, x, y, :, :].reshape([batch_size, 1, 32, 32])
    #             data_out = net.get_encoder_features(data_xy)
    #             spe_3D[:, :, x, y] = data_out.reshape([batch_size, 128])
    #     print(2)

    # spe_3D = torch.Tensor(np.zeros([batch_size, 128, 32, 32]))
    spe_3D = np.zeros([batch_size, 128, 32, 32])

    for sample in dataloader:
        data = sample['spe']
        flag = 0
        for i, each_path in enumerate(sample['path']):
            cate = each_path.split('/')[0]
            if not os.path.exists(save_path + cate):
                os.mkdir(save_path + cate)
            if not os.path.exists(save_path + each_path):
                flag = 1

        if flag:
            for y in range(32):
                data_xy = data[:, :, y, :, :].reshape([data.shape[0] * 32, 1, 32, 32])
                data_out = net.get_encoder_features(data_xy).cpu().data.reshape([data.shape[0], 32, 128])
                spe_3D[:, :, :, y] = np.transpose(data_out, (0,2,1))

                for i, each_path in enumerate(sample['path']):
                    if not os.path.exists(save_path + each_path):
                        np.save(save_path + each_path, spe_3D[i,:,:,:])

                        print(each_path)