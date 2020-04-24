# SAR_specific_models

1. This project provides some SAR specific models with strong abilities to extract spatial features of single-polarization Synthetic Aperture Radar (SAR) amplitude images.

2. A novel deep learning framework Deep SAR-Net (DSN) has been released for complex-valued SAR images. (code: DSN_src)

## Environment
Pytorch 0.4.0 (also verified in Pytorch 1.4.0)
Python 3.6

## SAR-Specific Models
./model/resnet18_I_nwpu_tsx.pth [1]
  
  The SAR image pre-trained model in Reference [1].

./model/resnet18_tsx_mstar_epoch7.pth [1]
  
  The transferred model to MSTAR 10-class target recognition task in Reference [1], achieving an overall accuracy of 99.46%.

./model/alexnet_tsx.pth [2]

  The SAR-image pre-trained model in Reference [2].

./model/alexnet_tsx_mstar_iter1920.pth [2]
  
  The transferred model to MSTAR 10-class target recognition task in Reference [2], achieving an overall accuracy of 99.34%.
  
./model/slc_spexy_cae_3.pth [3]

The pre-trained stacked convolutional auto-encoder model for frequency signals in Reference [3].

./model/slc_joint_deeper_3_F.pth [3]
  
  The trained DSN model in Reference [3].

## References
[1] Classification of Large-Scale High-Resolution SAR Images with Deep Transfer Learning, IEEE GRSL 2020

doi:  [10.1109/LGRS.2020.2965558](https://doi.org/10.1109/LGRS.2020.2965558) 

[2] What, Where and How to Transfer in SAR Target Recognition Based on Deep CNNs, IEEE TGRS 2019

doi:  [10.1109/TGRS.2019.2947634](https://doi.org/10.1109/TGRS.2019.2947634) 

[3] Deep SAR-Net: Learning Objects from Signals, submitted to ISPRS Journal of Photogrammetry and Remote Sensing 2020

doi:  [10.1016/j.isprsjprs.2020.01.016](http://doi.org/10.1016/j.isprsjprs.2020.01.016) 

## Deep SAR-Net (DSN) 

### Training Procedure
1. Run **data_process.py**

Generate the 4-D hyper-image signals of SAR images, and obtain the mean/std value for further usage.

2. Run **train_cae.py**

Train the stacked convolutional auto-encoder for frequency signals to obtain the cae model. 

3. Run **mapping_r4_r3.py**

To save the computing resources, map the 4-D hyper-image signal of SAR image to 3-D tensor with the pre-trained cae model.

4. Run **train_joint.py**

Train the post-learning subnet and fine-tuning the image representation subnet.

### Testing Procedure
1. Run **mapping_r4_r3.py**

Map the 4-D hyper-image signal of SAR images in test set.

2. Run **test_joint.py**

