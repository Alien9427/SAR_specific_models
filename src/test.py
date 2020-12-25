import torch
import tsx_dataset
from torch.utils.data import DataLoader
import network
import numpy as np
import torch.nn as nn
import transform_data
import argparse


def parameter_setting(args):
    config = {}

    config['data'] = {'nor_mean': 0.41900104848287917, 'nor_std': 0.06154960038356716, 'rescale_size': 160, 'crop_size': 128,
                      'data_root': 'D:/hzl/data/', 'data_txt': args.test_txt,
                      'batchsize': args.test_batchsize, 'data': args.data
                    }

    config['data']['test_crop_loc'] = int((config['data']['rescale_size'] - config['data']['crop_size'] - 1) / 2)


    config['model'] = {'cate_num': args.cate_num, 'pretrained_model_path': args.pretrained_model_path}

    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return config


def data_preparing(config):
    """1. data preparing """
    image_dataset = {}

    data_source = config['data']
    if data_source == 'tsx':
        data_test_transform = transform_data.data_transform_dict['tsx_test_5crop']
        data_transforms = data_test_transform(rescale_size=160, crop_size=128, channel=1)

        dataloaders = {}

        for i in range(5):
            image_dataset['test' + str(i)] = tsx_dataset.TSX_Dataset(txt_file=config['data_txt'],
                                                                     root_dir='D:/hzl/data/', data_type='npy',
                                                                     transform=data_transforms['val' + str(i)])
            dataloaders['test' + str(i)] = DataLoader(image_dataset['test' + str(i)], batch_size=config['batchsize'],
                                                      shuffle=False, num_workers=0)

        return dataloaders

    elif data_source == 'mstar':
        data_test_transform = transform_data.data_transform_dict['mstar_test']
        data_transforms = data_test_transform(rescale_size=128, crop_size=128, channel=3)
        image_dataset = tsx_dataset.Target_Dataset(txt_file=config['data_txt'], transform=data_transforms)

        return DataLoader(image_dataset,
                          batch_size=config['batchsize'],
                          shuffle=True,
                          num_workers=0)


    elif data_source == 'opensar':
        data_test_transform = transform_data.data_transform_dict['opensar']
        data_transforms = data_test_transform(channel=1)
        image_dataset = tsx_dataset.Target_Dataset(txt_file=config['data_txt'], transform=data_transforms)
        return DataLoader(image_dataset,
                          batch_size=config['batchsize'],
                          shuffle=True,
                          num_workers=0)



def model_preparing(config, config_data):
    if config_data['data'] == 'tsx':
        model = network.AlexNet_TSX(num_class=config['cate_num'])
    else:
        # model = network.AlexNet_OpenSAR(num_class=config['cate_num'])
        model = network.ResNet18_TSX(10)
    model.load_state_dict(torch.load(config['pretrained_model_path']))

    return model


def matrix_analysis(label_pred, label_true, cate_num):


    matrix = np.zeros([cate_num, cate_num])

    for i in range(cate_num):
        index = np.where(label_true == i)[0]
        label_term = label_pred[index]
        for j in range(cate_num):
            matrix[i, j] = len(np.where(label_term == j)[0])

    return matrix


def result_evaluation(matrix):

    label_num = matrix.shape[0]

    precision = [matrix[i, i] / sum(matrix[:, i]) for i in range(label_num)]
    recall = [matrix[i, i] / sum(matrix[i, :]) for i in range(label_num)]
    # f1_score = [2.0 / (1.0 / precision[i] + 1.0 / recall[i]) for i in range(label_num)]
    f1_score = [2.0 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(label_num)]

    return precision, recall, f1_score


def image_test_5crops(dataloader, model, config):
    acc_num = 0.0
    data_num = 0.0
    loss = 0.0
    cate_num = config['model']['cate_num']
    matrix = np.zeros([cate_num, cate_num])
    device = config['device']

    model.to(device)
    model.eval()

    if config['data']['data'] == 'tsx':
        iter_test = [iter(dataloader['test' + str(i)]) for i in range(5)]
        data_length = len(dataloader['test0'])
        for i in range(data_length):
            data = [iter_test[j].next() for j in range(5)]
            inputs = [data[j]['image'] for j in range(5)]
            labels = data[0]['label']
            for j in range(5):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)

            outputs = []
            outputs.append(model(inputs[j]))
            outputs = sum(outputs) / 5.0

            _, predict = torch.topk(outputs, 1, dim=1)
            acc_num += sum(labels.data.tolist()[i] in predict.data.tolist()[i] for i in range(len(labels.data)))
            data_num += labels.size()[0]
            class_criterion = nn.CrossEntropyLoss()
            loss += class_criterion(outputs, labels).item()
            matrix += matrix_analysis(torch.squeeze(predict).data.float(), labels.data.float(), cate_num)
    else:
        iter_test = iter(dataloader)
        data_length = len(dataloader)
        for i in range(data_length):
            data = iter_test.next()
            inputs = data['image']
            labels = data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predict = torch.topk(outputs, 1, dim=1)
            acc_num += torch.sum(torch.squeeze(predict) == labels.data).float()
            data_num += labels.size()[0]


            class_criterion = nn.CrossEntropyLoss()
            loss += class_criterion(outputs, labels).item()
            matrix += matrix_analysis(torch.squeeze(predict).data.float(), labels.data.float(), cate_num)

    """3. analyze the result"""

    if config['data']['data'] == 'tsx':
        dict_label_name = {0: 'Settlements', 1: 'Industrial_areas', 2: 'Public_transportations', 3: 'Agricultural_land',
                           4: 'Natural_vegetation',
                           5: 'Bare_ground', 6: 'Water_bodies'}
    elif config['data']['data'] == 'mstar':
        dict_label_name = {0:'2S1', 1:'BMP2', 2:'BRDM2', 3:'BTR60', 4:'BTR70', 5:'D7', 6:'T62', 7:'T72', 8:'ZIL131', 9:'ZSU234'}
    elif config['data']['data'] == 'opensar':
        dict_label_name = {0:'Bulk_Carrier', 1:'Cargo', 2:'Container_Ship'}

    # dict_label_name = {0: 'Settlements', 1: 'Industrial_areas', 2: 'Public_transportations', 3: 'Agricultural_land',
    #  4: 'Natural_vegetation', 5: 'Bare_ground', 6: 'Water_bodies', 7: 'Transportation_water', 8: 'Mountains'}
    # dict_label_name = {0: 'Public_transportations', 1: 'Agricultural_land', 2: 'Exotic_tree', 3: 'Water_bodies'}
    # dict_label_name = {0: 'Public_transportations', 1: 'Water_bodies'}

    print('overall accuracy:', (acc_num / data_num))
    print('loss:', loss / data_length)
    [p, r, f] = result_evaluation(matrix)
    print(matrix)
    for i in range(cate_num):
        print(dict_label_name[i], '\t\t', '{:.4f}'.format(p[i]), '{:.4f}'.format(r[i]), '{:.4f}'.format(f[i]))

    print('f1-score-avg:', sum(f) / cate_num)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test')
    parser.add_argument('--data', default='mstar')
    parser.add_argument('--test_txt', default='../data/mstar_val.txt')
    parser.add_argument('--test_batchsize', type=int, default=50)
    parser.add_argument('--cate_num', type=int, default=10)
    parser.add_argument('--pretrained_model_path', default='../model/resnet18_tsx_mstar_epoch7.pth')
    args = parser.parse_args()
    config = parameter_setting(args)

    dataloader = data_preparing(config['data'])
    models = model_preparing(config['model'], config['data'])
    print(config['device'])

    image_test_5crops(dataloader, models, config)
