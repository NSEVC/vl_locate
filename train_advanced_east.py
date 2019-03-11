# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : train_advanced_east.py
# @Time    : 2019/3/1 下午4:52
# @Desc    :

import os
import glob
import time
import cv2
import numpy as np
import tensorflow as tf

import config as cfg
from losses import tower_loss
from data import get_batch


def main():
    # input: images, label_score, label_maps
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_maps = tf.placeholder(tf.float32, shape=[None, None, None, 7], name='input_maps')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(cfg.learning_rate, global_step, decay_steps=3000, decay_rate=0.9, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate)

    reuse_variables = None
    total_loss, model_loss = tower_loss(input_images, input_maps, reuse_variables)
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
    # reuse_variables = True


    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(cfg.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')


    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(cfg.checkpoint_path, tf.get_default_graph())

    init_op = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # 1. restore or init
        if cfg.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(cfg.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init_op)


        # 2. generate data
        data_generator = get_batch(num_workers=cfg.num_readers,
                                   input_size=cfg.input_size,
                                   batch_size=cfg.batch_size
                                   )

        # 3. train loop
        start = time.time()
        for step in range(cfg.max_steps):
            data = next(data_generator)
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_maps: data[2]
                                                                                })
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * cfg.batch_size)/(time.time() - start)
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % cfg.save_checkpoint_steps == 0:
                saver.save(sess, cfg.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % cfg.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_maps: data[2]
                                                                                             })
            summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    main()

