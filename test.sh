#!/usr/bin/env bash
python src_joint/main.py test \
--dataset ptb \
--consttest-ptb-path data/23.auto.clean \
--deptest-ptb-path data/ptb_test_3.3.0.sd \
--embedding-path data/glove.gz \
--model-path-base models/joint_cwt_best_dev=93.85_devuas=95.87_devlas=94.47
