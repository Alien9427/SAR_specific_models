# &#x1F195; The Latest Version of DSN https://github.com/Alien9427/DSN

any questions please contact huangzhongling15@mails.ucas.ac.cn

```
@article{dsn2020,
title = {Deep SAR-Net: Learning objects from signals},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {161},
pages = {179-193},
year = {2020},
issn = {0924-2716},
author = {Z. Huang and M. Datcu and Z. Pan and B. Lei},
}
```


# SAR_specific_models

This project provides some SAR specific models with strong abilities to extract spatial features of single-polarization Synthetic Aperture Radar (SAR) amplitude images.


## Environment
Pytorch 0.4.0 (also verified in Pytorch 1.4.0)
Python 3.6

## SAR-Specific Models
./model/resnet18_I_nwpu_tsx.pth [1]
  
  The SAR image pre-trained model in Reference [1].
  
  It can be transferred to other SAR classification or detection models with ResNet-18 backbone.
  
  ### Transfer to Target Detection
  
  Use **MMDetection** to transfer the pretrained model to SAR detection:
  
  ![Picture1](https://user-images.githubusercontent.com/8330403/168396933-04780b94-a59c-4734-abc2-a4bd0d0c7834.png)


```
import torchvision.models as models
resnet18 = models.resnet18(pretrained=False)
pthfile = './model/resnet18_I_nwpu_tsx.pth'
resnet18.load_state_dict(torch.load(pthfile))
```

./model/resnet18_tsx_mstar_epoch7.pth [1]
  
  The transferred model to MSTAR 10-class target recognition task in Reference [1], achieving an overall accuracy of 99.46%.

./model/alexnet_tsx.pth [2]

  The SAR-image pre-trained model in Reference [2].

./model/alexnet_tsx_mstar_iter1920.pth [2]
  
  The transferred model to MSTAR 10-class target recognition task in Reference [2], achieving an overall accuracy of 99.34%.
 

## References
[1] Classification of Large-Scale High-Resolution SAR Images with Deep Transfer Learning, IEEE GRSL 2020

doi:  [10.1109/LGRS.2020.2965558](https://doi.org/10.1109/LGRS.2020.2965558) 

[2] What, Where and How to Transfer in SAR Target Recognition Based on Deep CNNs, IEEE TGRS 2019

doi:  [10.1109/TGRS.2019.2947634](https://doi.org/10.1109/TGRS.2019.2947634) 

