import argparse
import torch
import tsx_dataset
import transform_data
from torchvision import transforms
from torch.utils.data import DataLoader
import network
from train_or_test import alexnet_train, resnet_train
import torch.nn as nn
from sampler import ImbalancedDatasetSampler
import pandas as pd
from model_save_load import model_transfer


def parameter_setting(args):
    config = {}


    lr_layers_param = args.lr_layers

    optimizer_param = {
        'optim_type' : 'Adam', 'optim_param' : {'lr': 1.0, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.005},
        # 'optim_type' : 'SGD', 'optim_param' : {'lr': 1.0, 'momentum': 0.9, 'weight_decay': 0.005},
        'lr_type' : 'step', 'lr_param' : {'stepsize': 200000000, 'gamma': 0.1, 'init_lr': 1e-04},
        # 'lr_type' : 'multi_step', 'lr_param' : {'stepsize' : [2000, 3000, 4000], 'gamma' : 0.2, 'init_lr': 5e-06},
        # 'lr_type' : 'circle_multi_step', 'lr_param' : { 'stepsize' : [500, 1000, 3000, 4000],
        #                                                 'gamma' : [1e-2, 1e-3, 1e-4, 1e-5], 'init_lr': 1.0
                                                        # 'stepsize' : [500, 1000, 1500, 2500, 4000, 5000],
                                                        # 'gamma' : [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5], 'init_lr': 1.0
                                                    # },
        # 'lr_type' : 'inv', 'lr_param' : {"init_lr" : 1e-02, "gamma" : 0.0005, "power" : 0.75},
        # 'lr_layers' : {'conv1': 1, 'conv2': 1, 'conv3': 1, 'conv4': 1, 'conv5': 1, 'fc1': 1, 'fc2': 1}
        # 'lr_layers' : {'fc1':10}
    }




    config['data'] = {'source_data': args.source_data, 'target_data': args.target_data,
                      'data_txt': {'train': args.train_txt, 'val': args.val_txt},
                      'batchsize': {'train': args.train_batchsize, 'val': args.val_batchsize}
                     }

    config['model'] = {'cate_num': args.cate_num, 'save_model_path': args.save_model_path,
                       'transfer': args.transfer, 'transfer_layers': args.transfer_layers,
                       'pretrained_model_path': args.pretrained_model_path}

    cate_num = config['model']['cate_num']
    data_txt = config['data']['data_txt']
    data_all = open(data_txt['train'], 'r').readlines()
    data_dict = {line.split()[0]: line.split()[1] for line in data_all}
    train_count_dict = dict(pd.value_counts(list(data_dict.values())))
    loss_weight = [
        (1.0 - float(train_count_dict[str(i)]) / float(sum(train_count_dict.values()))) * cate_num / (cate_num - 1)
        for i in range(cate_num)]

    config['train_param'] = {'optimizer_param': optimizer_param, 'epoch_num': args.epoch_num}
    config['loss'] = {'loss_weight': loss_weight}
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if config['data']['target_data'] == 'tsx':
        optimizer_param['lr_layers'] = {'conv1': lr_layers_param[0], 'conv2': lr_layers_param[1],
                                        'conv3': lr_layers_param[2], 'conv4': lr_layers_param[3],
                                        'conv5': lr_layers_param[4], 'fc1': lr_layers_param[5], 'fc2': lr_layers_param[6]
                                        }
    else:
        optimizer_param['lr_layers'] = {'conv1': lr_layers_param[0], 'conv2': lr_layers_param[1],
                                        'conv3': lr_layers_param[2], 'conv4': lr_layers_param[3],
                                        'conv5': lr_layers_param[4], 'fc1': lr_layers_param[5]
                                        }
    return config


def data_preparing(config):
    """1. data preparing """
    data_source = config['target_data']
    if data_source == 'tsx':
        data_train_transform = transform_data.data_transform_dict['tsx_train']
        data_test_transform = transform_data.data_transform_dict['tsx_test']
        data_transforms = {
            'train': data_train_transform(rescale_size=160, crop_size=128, channel=1),
            'val': data_test_transform(rescale_size=160, crop_size=128, channel=1)
        }

        dataloaders = {}
        image_dataset = {
            x: tsx_dataset.TSX_Dataset(txt_file=config['data_txt'][x], root_dir='D:/hzl/data/', data_type='npy',
                                       transform=data_transforms[x])
            for x in ['train', 'val']
        }
        dataloaders['train'] = DataLoader(image_dataset['train'],
                                          batch_size=config['batchsize']['train'],
                                          # shuffle=True,
                                          sampler=ImbalancedDatasetSampler(image_dataset['train']),
                                          num_workers=0)
        dataloaders['val'] = DataLoader(image_dataset['val'],
                                        batch_size=config['batchsize']['val'],
                                        shuffle=True,
                                        num_workers=0)
        return dataloaders

    elif data_source == 'mstar':
        data_train_transform = transform_data.data_transform_dict['mstar_train']
        data_test_transform = transform_data.data_transform_dict['mstar_test']
        data_transforms = {
            'train': data_train_transform(rescale_size=128, crop_size=128, channel=3),
            'val': data_test_transform(rescale_size=128, crop_size=128, channel=3)
        }
        image_dataset = {
            x: tsx_dataset.Target_Dataset(txt_file=config['data_txt'][x], transform=data_transforms[x])
            for x in ['train', 'val']
        }
        return {
                x: DataLoader(image_dataset[x],
                              batch_size=config['batchsize'][x],
                              shuffle=True,
                              num_workers=0)
                for x in ['train', 'val']
        }

    elif data_source == 'opensar':
        data_transform_train_test = transform_data.data_transform_dict['opensar']
        data_transforms = {
            x: data_transform_train_test(channel=1)
            for x in ['train', 'val']
        }
        image_dataset = {
            x: tsx_dataset.Target_Dataset(txt_file=config['data_txt'][x], transform=data_transforms[x])
            for x in ['train', 'val']
        }
        return {
                x: DataLoader(image_dataset[x],
                              batch_size=config['batchsize'][x],
                              shuffle=True,
                              num_workers=0)
                for x in ['train', 'val']
        }



def model_preparing(config_model, config_data):
    source_data = config_data['source_data']
    target_data = config_data['target_data']

    if config_model['pretrained_model_path'] is None:
        if target_data == 'tsx':
            model = network.AlexNet_TSX(num_class=config_model['cate_num'])
        else:
            model = network.AlexNet_OpenSAR(num_class=config_model['cate_num'])
        return model

    elif config_model['transfer']:
        print('transfer')
        if source_data == 'tsx':
            # transferred_model = network.AlexNet_TSX(7)
            transferred_model = network.ResNet18_TSX(7)
        elif source_data == 'mstar':
            transferred_model = network.AlexNet_OpenSAR(10)
        elif source_data == 'imgnet':
            transferred_model = network.ResNet18()

        # model = network.AlexNet_OpenSAR(num_class=config_model['cate_num'])
        transferred_model.load_state_dict(torch.load(config_model['pretrained_model_path']))
        # num_feature = transferred_model.classifier[0].in_features

        num_feature = transferred_model.fc.in_features
        transferred_model.fc = nn.Linear(num_feature, config_model['cate_num'])
        transferred_model.fc.weight.data.normal_(0, 0.01)
        transferred_model.fc.bias.data.fill_(0.0)
        # model = model_transfer(model, transferred_model, config_model['transfer_layers'])

        # return model
        return transferred_model

    else:
        print('pretrained')
        if target_data == 'tsx':
            model = network.AlexNet_TSX(num_class=config_model['cate_num'])
        else:
            # model = network.AlexNet_OpenSAR(num_class=config_model['cate_num'])
            model = network.ResNet18_TSX(10)
        model.load_state_dict(torch.load(config_model['pretrained_model_path']))

        return model



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_test')
    parser.add_argument('--train_txt', default='../data/mstar_train.txt')
    parser.add_argument('--val_txt', default='../data/mstar_val.txt')
    parser.add_argument('--source_data', default='tsx')
    parser.add_argument('--target_data', default='mstar')
    parser.add_argument('--cate_num', type=int, default=10)
    parser.add_argument('--train_batchsize', type=int, default=128)
    parser.add_argument('--val_batchsize', type=int, default=100)
    parser.add_argument('--save_model_path', default='../model/resnet18_tsx_mstar_')
    parser.add_argument('--transfer', type=int, default=1)
    parser.add_argument('--transfer_layers', type=int, default=5)
    parser.add_argument('--pretrained_model_path', default='../model/resnet18_I_nwpu_tsx.pth.pth')
    parser.add_argument('--lr_layers', type=float, nargs='+', default=[0,0,0,0,0,1])
    parser.add_argument('--epoch_num', type=int, default=1000)

    args = parser.parse_args()
    config = parameter_setting(args)
    # print(config['data'])
    # print(config['model'])
    # print(config['loss'])
    # print(config['train_param'])
    dataloaders = data_preparing(config['data'])
    model = model_preparing(config['model'], config['data'])
    print(config['device'])
    resnet_train(dataloaders, model, config)


