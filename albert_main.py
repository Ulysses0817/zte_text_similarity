# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf_config = tf.ConfigProto()  
tf_config.gpu_options.allow_growth = True  
session = tf.Session(config=tf_config) 

import shutil, json

def create_data():
    """
    制作albert的训练集文件
    """
    # os.system("wget https://static.nowcoder.com/activity/2020zte/4/corpus.txt -O ./data/corpus.txt")
    for i in range(1, 10):
        print(os.listdir("./data"))
        os.system(f"python ./create_pretraining_data_sp.py --do_whole_word_mask=True --input_file=./data/corpus.txt \
            --output_file=./data/zte_textsim_{i}.tfrecord --vocab_file=./vocab.txt --do_lower_case=False \
            --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --non_chinese=True \
            --dupe_factor=1 --random_seed={i}")
        print(os.listdir("./data"))
    # return the score for hyperparameter tuning
    return 0

def pretrain():
    """
    训练集文件生成结束后，进行预训练
    """
    # os.system("gpustat -cpu")
    # os.system("cat /usr/local/cuda/version.txt")
    print(os.getcwd())
    opt_path = "./model/albert"
    print("save_path:", opt_path)
    
    ########### google albert #############  GPU(Google版本, small模型):

    os.system("python ./run_pretraining_google.py --input_file=./data/zte_textsim*.tfrecord  \
    --output_dir=%s --do_train=True --do_eval=True --albert_config_file=./albert_config_small_google.json \
    --train_batch_size=256 --max_seq_length=128 --max_predictions_per_seq=20 \
    --num_train_steps=250000 --num_warmup_steps=3125 --learning_rate=0.00176 \
    --save_checkpoints_steps=2000 --export_dir=%s/export "%(opt_path, opt_path))

    return 0

if __name__ == "__main__":
    create_data()
    pretrain()
