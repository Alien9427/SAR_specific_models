#!/usr/bin/env bash

python ./data_process.py
python ./train_cae.py
python ./mapping_r4_43.py --data_txt ../data/slc_train_3.txt
python ./mapping_r4_43.py --data_txt ../data/slc_val_3.txt
python ./train_joint.py
python ./test_joint.py
