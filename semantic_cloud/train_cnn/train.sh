#!/usr/bin/env sh
python train.py --dataset sunrgbd --img_rows 473 --img_cols 473 --n_epoch 50 --batch_size 2 --l_rate 0.001 --visdom
