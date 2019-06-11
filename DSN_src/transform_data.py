import torch
import numpy as np



class Numpy2Tensor(object):
    def __call__(self, data):
        return torch.Tensor(data)

class Normalize_spe(object):
    def __init__(self,
                 mean = np.array([3.76501112, 3.75611564]),
                 std = np.array([0.07338769, 0.06449222])):

        self.mean = mean
        self.std = std

    def __call__(self, data):
        for i in range(2):
            data[i] = (data[i] - self.mean[i]) / self.std[i]

        return data

class Normalize_spe_xy(object):
    def __init__(self,
                 max_value = 10.628257178154184,
                 min_value = 0.0011597341927439826):

        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, data):

        data = (data - self.min_value) / (self.max_value - self.min_value)

        return data


class Normalize_img(object):
    def __init__(self, mean=0.29982, std=0.07479776):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - self.mean) / self.std

        return img

class Numpy2Tensor_img(object):
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
            img_copy = torch.Tensor(img_copy)
            # backward compatibility
            return img_copy.float()