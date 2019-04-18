# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : predict.py
# @Time    : 2019/3/7 上午10:45
# @Desc    :

from __future__ import division, print_function

import os
import cv2
import time
import numpy as np
import tensorflow as tf

import model
from nms import nms
import config as cfg
from data import pad_image

import argparse
parser = argparse.ArgumentParser()
parser.parse_args()

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(cfg.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def visualize_activation_pixels(im_data, pred_map, activation_pixels, scale):
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        px = (j + 0.5) * cfg.pixel_size
        py = (i + 0.5) * cfg.pixel_size
        line_width, line_color = 1, (0, 0, 255)
        if pred_map[i, j, 1] >= cfg.side_vertex_pixel_threshold:
            if pred_map[i, j, 2] < cfg.trunc_threshold:  # 0.1
                line_width, line_color = 2, (255, 255, 0)
            elif pred_map[i, j, 2] >= 1 - cfg.trunc_threshold:
                line_width, line_color = 2, (0, 255, 0)

        pts = np.array([[px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size],
                        [px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size],
                        [px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size],
                        [px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size],
                        [px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size]]) / scale

        cv2.polylines(im_data, pts=np.int32([pts]), isClosed=True, color=line_color, thickness=line_width)

    return im_data

def visualize_out_box(im_data, quad_scores, quad_after_nms, scale, quiet=False):
    txt_items = []
    for score, geo, index in zip(quad_scores, quad_after_nms, range(len(quad_scores))):
        if np.amin(score) > 0:
            pts = np.array([geo[0], geo[1], geo[2], geo[3], geo[0]]) / scale
            cv2.polylines(im_data, pts=np.int32([pts]), isClosed=True, color=(0, 0, 255), thickness=2)

            rescaled_geo = geo / scale
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')

    return im_data, txt_items

def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    with tf.get_default_graph().as_default():
        # [1]. create graph
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_pred_map = model.model(input_images, is_training=False)

        # [2]. set train config
        variable_averages = tf.train.ExponentialMovingAverage(cfg.moving_average_decay, global_step)

        # [3]. create saver and restore
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(cfg.checkpoint_path)   # return: CheckpointState proto file or None
            model_path = os.path.join(cfg.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            # [4]. get image and predict
            image_list = get_images()
            for image_file in image_list:
                im_data = cv2.imread(image_file)
                im_data_padded, scale = pad_image(im_data, cfg.model_height, cfg.model_weight)

                timer = {'net': 0, 'nms': 0}
                start = time.time()
                pred_map = sess.run(f_pred_map, feed_dict={input_images: [im_data_padded]})
                timer['net'] = time.time() - start

                # [5]. post process
                pred_map = np.squeeze(pred_map, axis=0)
                pred_map[:, :, :3] = sigmoid(pred_map[:, :, :3])
                cv2.imwrite(os.path.join(cfg.test_output_dir, image_file.split('/')[-1].split('.')[0] + '_map.jpg'), pred_map[:, :, :1] * 255)

                # get activation pixels
                cond = np.greater_equal(pred_map[:, :, 0], cfg.pixel_threshold)
                activation_pixels = np.where(cond)

                # get quad after nms; nms
                start = time.time()
                quad_scores, quad_after_nms = nms(pred_map, activation_pixels)
                timer['nms'] = time.time() - start

                print('Time cost:', 'net predict %f' % timer['net'], 'nms %f' % timer['nms'])

                # [6]. visualize
                if cfg.visualization:
                    # visualize the activation pixels
                    im_data_act = visualize_activation_pixels(im_data.copy(), pred_map, activation_pixels, scale)
                    cv2.imwrite(os.path.join(cfg.test_output_dir, image_file.split('/')[-1].split('.')[0] + '_act.jpg'), im_data_act)

                    # visualize the out boxes
                    im_data_detect, txt_items = visualize_out_box(im_data.copy(), quad_scores, quad_after_nms, scale)
                    cv2.imwrite(os.path.join(cfg.test_output_dir, image_file.split('/')[-1].split('.')[0] + '_out.jpg'), im_data_detect)

                    if cfg.predict_write2txt and len(txt_items) > 0:
                        with open(image_file.split('.')[0] + '.txt', 'w') as f_txt:
                            f_txt.writelines(txt_items)

if __name__ == '__main__':
    main()
