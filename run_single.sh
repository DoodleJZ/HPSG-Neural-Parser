#!/usr/bin/env bash
python src_joint/main.py train \
 --model-path-base models/joint_single \
 --epochs 150 \
 --use-chars-lstm \
 --use-words \
 --use-tags \
 --use-cat \
 --const-lada 0.5 \
 --num-layers 12 \
 --dataset ptb \
 --embedding-path data/glove.gz \
 --model-name joint_single \
 --embedding-type glove \
 --checks-per-epoch 4 \
 --train-ptb-path data/02-21.10way.clean \
 --dev-ptb-path data/22.auto.clean \
 --dep-train-ptb-path data/ptb_train_3.3.0.sd \
 --dep-dev-ptb-path data/ptb_dev_3.3.0.sd