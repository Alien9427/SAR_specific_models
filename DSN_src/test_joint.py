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
import pandas as pd

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


def funtest_prf_mtx(dataloader):
    acc_num = 0.0
    data_num = 0
    matrix = np.zeros([cate_num, cate_num])
    iter_val = iter(dataloader)
    for j in range(len(dataloader)):
        val_data = next(iter_val)
        val_img = val_data['img'].to(device)
        val_spe = val_data['spe'].to(device)
        val_label = val_data['label'].to(device)
        val_output = net_joint(val_spe, val_img)

        # val_loss = loss_func(val_output, val_label)
        _, pred = torch.Tensor.max(val_output, 1)
        acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
        data_num += val_label.size()[0]
        matrix += matrix_analysis(torch.squeeze(pred).data.float(), val_label.data.float(), cate_num)

        # val_loss += loss_func(val_output, val_label).item()

    # val_loss /= len(dataloader)
    val_acc = acc_num / data_num
    [p, r, f] = result_evaluation(matrix)
    print(val_acc)
    for i in range(cate_num):
        print(label2name.loc[i]['catename'], '\t\t\t\t', '{:.4f}'.format(p[i]), '{:.4f}'.format(r[i]),
              '{:.4f}'.format(f[i]))
    print('f1-score-avg:', sum(f) / cate_num)
    print(matrix)


def save4roc(dataloader):
    y_score = np.zeros([0,8])
    y_label = np.zeros([0], dtype=int)
    iter_val = iter(dataloader)
    for j in range(len(dataloader)):
        val_data = next(iter_val)
        val_img = val_data['img'].to(device)
        val_spe = val_data['spe'].to(device)
        val_label = val_data['label'].to(device)
        val_output = net_joint(val_spe, val_img)
        y_label = np.concatenate((y_label, val_label.cpu().data.numpy()), axis=0)
        y_score = np.concatenate((y_score, val_output.cpu().data.numpy()))

    return pd.DataFrame(data=y_score, index=y_label)


def save4feature(dataloader, net):
    global y_feature
    y_feature = np.zeros([0,512])
    y_label = np.zeros([0], dtype=int)
    y_path = []
    iter_val = iter(dataloader)

    def hook_features(module, input, output):
        global y_feature
        y_feature = np.concatenate((y_feature, output.cpu().data.numpy().squeeze()))

    net._modules.get('post_slc').register_forward_hook(hook_features)

    for j in range(len(dataloader)):
        val_data = next(iter_val)
        val_img = val_data['img'].to(device)
        val_spe = val_data['spe'].to(device)
        val_label = val_data['label'].to(device)
        val_path = val_data['path']
        net(val_spe, val_img)
        y_label = np.concatenate((y_label, val_label.cpu().data.numpy()), axis=0)
        y_path = y_path + val_path

    df_return = pd.DataFrame(data=y_feature, index=y_label)
    df_return['path'] = pd.Series(y_path, index=y_label)

    return df_return



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt_file = '../data/slc_val_19.txt'
    batch_size = 30
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



    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    pretrained_model = '../model/slc_joint_deeper_19_F.pth'
    net_joint = network.SLC_joint2(cate_num)
    net_joint.load_state_dict(torch.load(pretrained_model))

    net_joint.to(device)


    i = 0


    label2name = pd.read_csv('../data/catename2label_cate8.txt')

    net_joint.eval()

    # funtest_prf_mtx(dataloader)
    df_score = save4roc(dataloader)
    model_name = pretrained_model.split('/')[-1].split('.')[0]
    df_score.to_csv('../results/' + model_name + '_df.txt', index_label='label')
    # df_feature = save4feature(dataloader, net_joint)
    # model_name = pretrained_model.split('/')[-1].split('.')[0]
    # df_feature.to_csv('../results/' + model_name + '_df_feature.txt', index_label='label')