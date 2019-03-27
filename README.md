# SAR_specific_models

This project provides some SAR specific models which have a strong ability to extract spatial features of single-polarization Synthetic Aperture Radar images.

## reference
[1] Imbalanced Large-Scale Complex Land Cover Classification of High-Resolution SAR Images with Deep Transfer Learning (undergoing review)

[2] What, Where and How to Transfer in SAR Target Recognition Based on Deep CNNs (undergoing review)

preprint version https://www.researchgate.net/publication/331997369_What_Where_and_How_to_Transfer_in_SAR_Target_Recognition_Based_on_Deep_CNNs

## Environment
pytorch 0.4.0
python 3.6

## SAR specific model zoo
./models/resnet18_I_nwpu_cate45_tsx_level1_cate7_col36_imb_ce+topk+.pth
  
  see ref[1]

./models/resnet18_tsx_mstar_epoch7.pth
  
  see ref[1] achieving an overall accuracy of 99.46% on MSTAR 10-class target recognition task

./models/alexnet_tsx.pth

  see ref[2]
