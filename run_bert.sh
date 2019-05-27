#!/usr/bin/env bash
python src_joint/main.py train \
 --model-path-base models/joint_bert \
  --epochs 100 \
 --use-bert \
 --const-lada 0.8 \
 --dataset ptb \
 --embedding-path data/glove.gz \
 --model-name joint_bert \
 --checks-per-epoch 4 \
 --num-layers 2 \
 --learning-rate 0.00005 \
 --batch-size 50 \
 --eval-batch-size 20 \
 --subbatch-max-tokens 500 \
 --train-ptb-path data/02-21.10way.clean \
 --dev-ptb-path data/22.auto.clean \
 --dep-train-ptb-path data/ptb_train_3.3.0.sd \
 --dep-dev-ptb-path data/ptb_dev_3.3.0.sd