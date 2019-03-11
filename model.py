# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : model.py
# @Time    : 2019/3/5 下午8:19
# @Desc    : advancedEAST model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

import resnet_utils
import resnet_v1

resnet_arg_scope = resnet_utils.resnet_arg_scope


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map

            f_score = slim.conv2d(g[3], 1, 1, activation_fn=None, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            side_vertex_code_map = slim.conv2d(g[3], 2, 1, activation_fn=None, normalizer_fn=None)
            side_vertex_coord_map = (slim.conv2d(g[3], 4, 1, activation_fn=None, normalizer_fn=None) - 0.5)

            out_map = tf.concat([f_score, side_vertex_code_map, side_vertex_coord_map], axis=-1)

    return out_map

