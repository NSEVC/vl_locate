# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : config.py
# @Time    : 2019/3/5 下午4:15
# @Desc    :

from __future__ import division, print_function, absolute_import

import os


root_path = '/share_sdb/clj/vl_locate'
# ==== load data ====
dataset_name = 'vehicle_license'   # ['vehicle_license', 'namplate_vin']
model_height = 416
model_weight = 576
channel = 3
batch_size = 16
num_readers = 10

pixel_size = 4

# ==== data preprocess ====
min_crop_side_ratio = 0.7
min_text_size = 10   # 文字框最小的边长
shrink_ratio = 0.3
shrink_side_ratio = 0.6
PIXEL_MEAN = [123.68, 116.779, 103.939]

# ==== loss ====
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0


# ==== train ====
restore = True
moving_average_decay = 0.9
learning_rate = 1e-2
epsilon = 1e-4
epoch = 3
max_steps = 100000
save_checkpoint_steps = 1000
save_summary = True
save_summary_steps = 100


# ==== predict ====
pixel_threshold = 0.8
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
visualization = False
predict_write2txt = False



def makedirs(path):
    if not os.path.exists(path):
        os.mkdir(path)

training_data_path = os.path.join(root_path, 'data/train', dataset_name)
makedirs(training_data_path)

test_data_path = os.path.join(root_path, 'data/test', dataset_name)
makedirs(test_data_path)

test_output_dir = os.path.join(root_path, 'data/test', 'out_'+dataset_name)
makedirs(test_output_dir)

checkpoint_path = os.path.join(root_path, 'backup', 'backup_'+dataset_name)
makedirs(checkpoint_path)



