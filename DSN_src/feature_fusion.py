import network
import transform_data
import slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ImbalancedDatasetSampler
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from learning_schedule import param_setting_jointmodel2
from torch import optim
import dcaFuse

def get_pretrained(img_model, net_joint):
    # spe_mapping = {'encoder1':'pre_spe.0',
    #                'encoder2':'pre_spe.2',
    #                'encoder3':'pre_spe.3',
    #                'encoder4':'pre_spe.5'}
    img_mapping = {'conv1':'pre_img.0',
                   'bn1':'pre_img.1',
                   'layer1':'pre_img.3',
                   'layer2':'pre_img.4'}
    # for key in spe_model.keys():
    #     if key[:8] in spe_mapping.keys():
    #         net_joint.state_dict()[spe_mapping[key[:8]] + key[8:]].data.copy_(spe_model[key])
    for key in img_model.keys():
        if key[:5] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:5]] + key[5:]].data.copy_(img_model[key])
        elif key[:3] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:3]] + key[3:]].data.copy_(img_model[key])
        elif key[:6] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:6]] + key[6:]].data.copy_(img_model[key])
    return net_joint


def get_train_features_transmat(data_dir):
    # data_dir = 'slc_train_3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    txt_file = '../data/' + data_dir + '.txt'

    batch_size = 10
    cate_num = 8

    spe_transform = transforms.Compose([
        # transform_data.Normalize_spe_xy(),
        transform_data.Numpy2Tensor()
    ])

    img_transform = transforms.Compose([
        transform_data.Normalize_img(),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = slc_dataset.SLC_img_spe4D(txt_file=txt_file,
                                        img_dir='../data/slc_data/',
                                        spe_dir='../data/spexy_data_3/',
                                        img_transform=img_transform,
                                        spe_transform=spe_transform)

    dataloaders = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    img_model = torch.load('../model/tsx.pth')
    net_joint = network.SLC_joint2(cate_num)
    net_joint = get_pretrained(img_model, net_joint)

    net_joint.to(device)

    np_label = np.zeros(0)
    np_img_feature = np.zeros([0, 128, 16, 16])
    np_spe_feature = np.zeros([0, 128, 16, 16])

    for data in dataloaders:
        img_data = data['img'].to(device)
        spe_data = data['spe'].to(device)
        labels = data['label'].to(device)
        img_features = net_joint.pre_img_features(img_data)
        spe_features = net_joint.pre_spe_features(spe_data)

        np_label = np.concatenate((np_label, labels.cpu().data.numpy()))
        np_img_feature = np.concatenate((np_img_feature, img_features.cpu().data.numpy()), axis=0)
        np_spe_feature = np.concatenate((np_spe_feature, spe_features.cpu().data.numpy()), axis=0)

    N = len(np_label)
    np_img_feature = np_img_feature.reshape([N, 128, 256])
    np_spe_feature = np_spe_feature.reshape([N, 128, 256])

    transmat_img_feature = np.zeros([0, 7, 128])
    transmat_spe_feature = np.zeros([0, 7, 128])
    output_img_feature = np.zeros([0, 7, N])
    output_spe_feature = np.zeros([0, 7, N])

    # _, _, x, y = img_features.shape
    for i in range(256):
        per_img_feature = np_img_feature[:, :, i].T
        per_spe_feature = np_spe_feature[:, :, i].T
        per_output_img_feature, per_output_spe_feature, per_transmat_img_feature, per_transmat_spe_feature = \
            dcaFuse.dcaFuse(per_img_feature, per_spe_feature, np_label)

        transmat_img_feature = np.concatenate((transmat_img_feature, per_transmat_img_feature.reshape([1, 7, 128])),
                                              axis=0)
        transmat_spe_feature = np.concatenate((transmat_spe_feature, per_transmat_spe_feature.reshape([1, 7, 128])),
                                              axis=0)
        output_img_feature = np.concatenate((output_img_feature, per_output_img_feature.reshape([1, 7, N])), axis=0)
        output_spe_feature = np.concatenate((output_spe_feature, per_output_spe_feature.reshape([1, 7, N])), axis=0)

    np.save('../data/' + data_dir + '_img_features.npy',
            output_img_feature.reshape([16, 16, 7, N]).transpose(3, 2, 0, 1))
    np.save('../data/' + data_dir + '_spe_features.npy',
            output_spe_feature.reshape([16, 16, 7, N]).transpose(3, 2, 0, 1))
    np.save('../data/' + data_dir + '_transmat_img.npy', transmat_img_feature.reshape([16, 16, 7, 128]))
    np.save('../data/' + data_dir + '_transmat_spe.npy', transmat_spe_feature.reshape([16, 16, 7, 128]))
    np.save('../data/' + data_dir + '_label.npy', np_label)



def get_val_features(data_dir, transmat_img_dir, transmat_spe_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    txt_file = '../data/' + data_dir + '.txt'

    batch_size = 10
    cate_num = 8

    spe_transform = transforms.Compose([
        # transform_data.Normalize_spe_xy(),
        transform_data.Numpy2Tensor()
    ])

    img_transform = transforms.Compose([
        transform_data.Normalize_img(),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = slc_dataset.SLC_img_spe4D(txt_file=txt_file,
                                        img_dir='../data/slc_data/',
                                        spe_dir='../data/spexy_data_3/',
                                        img_transform=img_transform,
                                        spe_transform=spe_transform)

    dataloaders = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    img_model = torch.load('../model/tsx.pth')
    net_joint = network.SLC_joint2(cate_num)
    net_joint = get_pretrained(img_model, net_joint)

    net_joint.to(device)

    np_label = np.zeros(0)
    np_img_feature = np.zeros([0, 128, 16, 16])
    np_spe_feature = np.zeros([0, 128, 16, 16])

    for data in dataloaders:
        img_data = data['img'].to(device)
        spe_data = data['spe'].to(device)
        labels = data['label'].to(device)
        img_features = net_joint.pre_img_features(img_data)
        spe_features = net_joint.pre_spe_features(spe_data)

        np_label = np.concatenate((np_label, labels.cpu().data.numpy()))
        np_img_feature = np.concatenate((np_img_feature, img_features.cpu().data.numpy()), axis=0)
        np_spe_feature = np.concatenate((np_spe_feature, spe_features.cpu().data.numpy()), axis=0)

    N = len(np_label)
    # np_img_feature = np_img_feature.reshape([N, 128, 256])
    # np_spe_feature = np_spe_feature.reshape([N, 128, 256])

    transmat_img = np.load(transmat_img_dir)
    transmat_spe = np.load(transmat_spe_dir)

    output_img_feature = np.zeros([16, 16, 7, N])
    output_spe_feature = np.zeros([16, 16, 7, N])

    for i in range(16):
        for j in range(16):
            per_img_feature = np_img_feature[:, :, i, j].T
            per_spe_feature = np_spe_feature[:, :, i, j].T
            per_output_img_feature = np.dot(np.squeeze(transmat_img[i, j, :, :]), per_img_feature)
            per_output_spe_feature = np.dot(np.squeeze(transmat_spe[i, j, :, :]), per_spe_feature)

            output_img_feature[i, j, :, :] = per_output_img_feature
            output_spe_feature[i, j, :, :] = per_output_spe_feature

    np.save('../data/' + data_dir + '_img_features.npy',
            output_img_feature.transpose([3, 2, 0, 1]))
    np.save('../data/' + data_dir + '_spe_features.npy',
            output_spe_feature.transpose([3, 2, 0, 1]))
    np.save('../data/' + data_dir + '_label.npy',
            np_label)


if __name__ == '__main__':
    get_train_features_transmat(data_dir='../data/slc_train_3')
    get_val_features(data_dir='../data/slc_val_3',
                     transmat_img_dir='../data/slc_train_3_transmat_img.npy',
                     transmat_spe_dir='../data/slc_train_3_transmat_spe.npy')





