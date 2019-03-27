import numpy as np
import cv2
import random
import torch
from torchvision import transforms

"""
These callable classes are all for SAR images which have only 1 channel and read as np.array [0, 1]
"""


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - self.mean) / self.std

        return img

class Rescale(object):

    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size

    def __call__(self, img):
      th, tw = self.size
      return cv2.resize(img, (th, tw))

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        img_h, img_w = img.shape
        start_x = random.randint(0,img_w-tw)
        start_y = random.randint(0,img_h-th)
        return img[start_y:start_y+th, start_x:start_x+tw]

class PlaceCrop(object):
    """Crops the given np.array at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (np.array): Image to be cropped.
        Returns:
            np.array: Cropped image.
        """
        th, tw = self.size
        return img[self.start_y : self.start_y + th, self.start_x : self.start_x + tw]

class RandomFlip(object):
    """Horizontally flip the given np.array randomly with a probability of 0.5 """

    def __call__(self, img):
        flag = random.randint(0,1)
        if flag:
            return img[:, ::-1]
        else:
            return img

class LogTransform(object):
    def __call__(self, img):
        return np.log2(np.double(img + 1)) / 16

class Numpy2Tensor(object):
    """Convert a 1-channel ``numpy.ndarray`` to 1-c or 3-c tensor,
    depending on the arg parameter of "channels"
    """
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, img):
        """
        for SAR images (.npy), shape H * W, we should transform into C * H * W
        :param img:
        :return:
        """
        channels = self.channels
        img_copy = np.zeros([channels, img.shape[0], img.shape[1]])

        for i in range(channels):
            img_copy[i, :, :] = np.reshape(img, [1, img.shape[0], img.shape[1]]).copy()

        if not isinstance(img_copy, np.ndarray) and (img_copy.ndim in {2, 3}):
            raise TypeError('img should be ndarray. Got {}'.format(type(img_copy)))

        if isinstance(img_copy, np.ndarray):
            # handle numpy array
            img_copy = torch.from_numpy(img_copy)
            # backward compatibility
            return img_copy.float()

def tsx_test_5crop(rescale_size, crop_size, channel):

  #ten crops for image when validation, input the data_transforms dictionary
  nor_mean = 0.41900104848287917
  nor_std = 0.06154960038356716
  start_first = 0
  start_center = int((rescale_size - crop_size - 1) / 2)
  start_last = rescale_size - crop_size - 1
  data_transforms = {}
  data_transforms['val0'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(rescale_size),
      PlaceCrop(crop_size, start_first, start_first),
      Numpy2Tensor(channel)
  ])
  data_transforms['val1'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(rescale_size),
      PlaceCrop(crop_size, start_last, start_last),
      Numpy2Tensor(channel)
  ])
  data_transforms['val2'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(rescale_size),
      PlaceCrop(crop_size, start_last, start_first),
      Numpy2Tensor(channel)
  ])
  data_transforms['val3'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(rescale_size),
      PlaceCrop(crop_size, start_first, start_last),
      Numpy2Tensor(channel)
  ])
  data_transforms['val4'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(rescale_size),
      PlaceCrop(crop_size, start_center, start_center),
      Numpy2Tensor(channel)
  ])

  return data_transforms

def tsx_test(rescale_size, crop_size, channel):
    nor_mean = 0.41900104848287917
    nor_std = 0.06154960038356716
    test_crop_loc = int((rescale_size - crop_size - 1) / 2)

    return transforms.Compose([
            LogTransform(),
            Normalize(nor_mean, nor_std),
            Rescale(rescale_size),
            PlaceCrop(crop_size, test_crop_loc, test_crop_loc),
            Numpy2Tensor(channel)
        ])



def tsx_train(rescale_size, crop_size, channel):
    nor_mean = 0.41900104848287917
    nor_std = 0.06154960038356716
    return transforms.Compose([
                            LogTransform(),
                            Normalize(nor_mean, nor_std),
                            Rescale(rescale_size),
                            RandomCrop(crop_size),
                            RandomFlip(),
                            Numpy2Tensor(channel)
                            ])

def mstar_train(rescale_size, crop_size, channel):
    nor_mean =  0.0293905372581
    nor_std = 0.0308426998737
    return transforms.Compose([
        Normalize(nor_mean, nor_std),
        Rescale(rescale_size),
        RandomCrop(crop_size),
        RandomFlip(),
        Numpy2Tensor(channel)
    ])

def mstar_test(rescale_size, crop_size, channel):
    nor_mean = 0.0293905372581
    nor_std = 0.0308426998737
    test_crop_loc = int((rescale_size - crop_size - 1) / 2)
    return transforms.Compose([
        Normalize(nor_mean, nor_std),
        Rescale(rescale_size),
        PlaceCrop(crop_size, test_crop_loc, test_crop_loc),
        Numpy2Tensor(channel)
    ])

def opensar(channel):
    nor_mean = 0.031889471986319175
    nor_std = 0.050334902771837366
    return transforms.Compose([
        Normalize(nor_mean, nor_std),
        Numpy2Tensor(channel)
    ])


data_transform_dict = {'tsx_train': tsx_train, 'tsx_test': tsx_test, 'tsx_test_5crop': tsx_test_5crop,
                       'mstar_train': mstar_train, 'mstar_test': mstar_test,
                       'opensar': opensar}