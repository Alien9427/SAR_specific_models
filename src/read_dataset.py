import numpy as np
import cv2
import os


def read_dataset_txt(txt_file):
    list_path_label = {}
    f_txt = open(txt_file)
    lines_txt = f_txt.readlines()

    for count, l in enumerate(lines_txt):
        l_split = l.split()
        l_path = l_split[0]
        l_label = l_split[1]

        list_path_label[count] = [l_path, int(l_label)]

    return list_path_label


def read_patch_0116(prefix_path, patch_path):

    collection_name = patch_path.split('/')[0]
    patch_name = patch_path.split('/')[-1]

    scene_name = patch_name[4:63]
    image_name = patch_name[64:85] + '.tif'

    if patch_name[:3] == 'QLK':
        [pixel, loc_x, loc_y] = patch_name[86:-4].split('_')

        pixel = int(pixel); loc_x = int(loc_x); loc_y = int(loc_y)

        image_path = os.path.join(collection_name, scene_name, 'IMAGEDATA', image_name)
        # image = cv2.imread(os.path.join(prefix_path, image_path), cv2.IMREAD_ANYDEPTH)
        image = np.double(cv2.imread(os.path.join(prefix_path, image_path), cv2.IMREAD_ANYDEPTH)) / 65535.0
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # [w, h] = image.shape
        patch_image = image[pixel * loc_x : pixel * (loc_x + 1), pixel * loc_y : pixel * (loc_y + 1)]

        # cv2.imshow('patch_image', patch_image)
        # cv2.waitKey(0)

        return patch_image

    # a question about how to normalize the patch


def read_patch_1737(prefix_path, patch_path):

    dict_num_imagename = {}

    """dict_num_imagename = {'CollectionXX' : {1 : image_path, 2 : image_path, ...}, 'CollectionXX' : {}, ...}
            for each Collection, mapping the "image_num" in patches and imagepath
    """
    for i in range(17, 38):
        dict_num_imagename['Collection' + str(i)] = {}
        image_list = sorted(os.listdir(prefix_path + '/Collection' + str(i)))
        for count, item in enumerate(image_list):
            if item[0] != '_':
                imagedata_list = os.listdir(prefix_path + '/' + os.path.join('Collection' + str(i), item, 'IMAGEDATA'))
                for imagedata in imagedata_list:
                    if imagedata[-7:-4] == 'cut' and imagedata[0] != '.':
                        dict_num_imagename['Collection' + str(i)][count+1] = os.path.join('Collection' + str(i), item,
                                                                                          'IMAGEDATA', imagedata)
                        break
                    else:
                        dict_num_imagename['Collection' + str(i)][count+1] = os.path.join('Collection' + str(i), item,
                                                                                          'IMAGEDATA', imagedata)

    # print dict_num_imagename['Collection30']

    pixel = 160

    collection_name = patch_path.split('/')[0]
    patch_name = patch_path.split('/')[-1]

    [image_num, loc] = patch_name[:-4].split('_')
    image_path = dict_num_imagename[collection_name][int(image_num)]
    # image = cv2.imread(os.path.join(prefix_path, image_path), cv2.IMREAD_ANYDEPTH)
    image = np.double(cv2.imread(os.path.join(prefix_path, image_path), cv2.IMREAD_ANYDEPTH)) / 65535.0

    [h, w] = image.shape
    each_row = np.floor(np.double(w) / np.double(pixel))
    loc_y = int(float(loc) % each_row) - 1
    loc_x = int(np.floor(float(loc) / each_row))
    if loc_y == -1:
        loc_y = int(each_row) - 1
        loc_x -= 1
    patch_image = image[pixel * loc_x: pixel * (loc_x + 1), pixel * loc_y: pixel * (loc_y + 1)]

    # patch_image = (patch_image - np.min(patch_image)) / (np.max(patch_image) - np.min(patch_image))
    # cv2.imshow('patch_image', patch_image)
    # cv2.waitKey(0)

    return patch_image

def read_npy(patch_path):
    # patch = np.double(np.load(patch_path)) / 65535.0
    patch = np.load(patch_path)

    return patch

def read_tif(prefix_path, patch_path):
    collection_num = int(patch_path.split('Collection')[1][:2])
    if collection_num <= 16:
        patch = read_patch_0116(prefix_path, patch_path)
    else:
        patch = read_patch_1737(prefix_path, patch_path)

    return patch

def read_jpg(patch_path):
    img_size = 160
    patch = np.double(cv2.imread(patch_path)[:,:,0].reshape([img_size,img_size])) / 255.0

    return patch

def read_sar_target(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img.dtype == 'uint16':
        return img / 65535.0
    elif img.dtype == 'uint8':
        return img / 255.0
