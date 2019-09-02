#!/usr/bin/env bash
python src_joint/main.py train \
 --model-path-base models/joint_xlnet \
  --epochs 100 \
 --use-xlnet \
 --const-lada 0.8 \
 --dataset ptb \
 --embedding-path data/glove.gz \
 --model-name joint_xlnet \
 --checks-per-epoch 4 \
 --num-layers 2 \
 --learning-rate 0.00005 \
 --batch-size 100 \
 --eval-batch-size 20 \
 --subbatch-max-tokens 1000 \
 --train-ptb-path data/02-21.10way.clean \
 --dev-ptb-path data/22.auto.clean \
 --dep-train-ptb-path data/ptb_train_3.3.0.sd \
 --dep-dev-ptb-path data/ptb_dev_3.3.0.sd