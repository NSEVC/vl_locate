# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : predict.py
# @Time    : 2019/3/7 上午10:45
# @Desc    :

import os
import cv2
import time
import numpy as np
import tensorflow as tf

import model
from nms import nms
import config as cfg


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


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    try:
        os.makedirs(cfg.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_pred_map = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(cfg.checkpoint_path)
            model_path = os.path.join(cfg.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            image_list = get_images()
            for image_file in image_list:
                im_data = cv2.imread(image_file)[:, :, ::-1]
                im_resized, (ratio_h, ratio_w) = resize_image(im_data)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                pred_map = sess.run([f_pred_map], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start
                print('predict cost:', timer['net'])

                pred_map = np.squeeze(pred_map, axis=0)
                pred_map[:, :, :3] = sigmoid(pred_map[:, :, :3])
                # filter the score map
                cond = np.greater_equal(pred_map[:, :, 0], cfg.pixel_threshold)
                activation_pixels = np.where(cond)

                quad_scores, quad_after_nms = nms(pred_map, activation_pixels)

                quad_im = im_resized.copy()
                for i, j in zip(activation_pixels[0], activation_pixels[1]):
                    px = (j + 0.5) * cfg.pixel_size
                    py = (i + 0.5) * cfg.pixel_size
                    line_width, line_color = 1, (0, 0, 255)
                    if pred_map[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                        if pred_map[i, j, 2] < cfg.trunc_threshold:  # 0.1
                            line_width, line_color = 2, (255, 255, 0)
                        elif pred_map[i, j, 2] >= 1 - cfg.trunc_threshold:
                            line_width, line_color = 2, (0, 255, 0)
                    cv2.polylines(quad_im,
                                  pts=np.array([[px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size],
                                               [px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size],
                                               [px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size],
                                               [px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size],
                                               [px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size]]),
                                  isClosed=True, color=line_color, thickness=line_width
                                  )

                cv2.imwrite(os.path.join(cfg.output_dir, image_file.split('/')[-1].split('.')[0] + '_act.jpg'), quad_im)


if __name__ == '__main__':
    main()
