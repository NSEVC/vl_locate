# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : config.py
# @Time    : 2019/3/5 下午4:15
# @Desc    :


# ==== load data ====
# model_size_h = 416
# model_size_w = 576
training_data_path = '/share_sdb/clj/vl_locate/data/train/vl_3'

input_size = 512
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
save_summary_steps = 100

checkpoint_path = '/share_sdb/clj/vl_locate/tmp/'


# ==== predict ====
test_data_path = '/share_sdb/clj/vl_locate/data/test/JPEGImages'
output_dir = '/share_sdb/clj/vl_locate/data/test/out'
max_predict_img_size = input_size
pixel_threshold = 0.8
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1



