#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import DataGenerator
from actor import Actor
from config import get_config, print_config


def main():
    # Get running configuration
    config, _ = get_config()
    print_config()

    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config)

    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

    rewards = []

    result_pos_list = []

    print("Starting session...")
    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        if config.restore_model is True:
            saver.restore(sess, config.restore_from)
            print("Model restored.")

        training_set = DataGenerator(config)

        # training mode
        if config.training_mode:

            print("Starting training...")
            for i in tqdm(range(config.iteration)):
                # Get feed dict
                input_batch = training_set.train_batch()
                feed = {actor.input_: input_batch}

                # Forward pass & train step
                positions, reward, train_step1, train_step2 = sess.run(
                    [actor.positions, actor.reward, actor.train_step1, actor.train_step2],
                    feed_dict=feed)

                rewards.append(np.mean(reward))

                if i % 100 == 0 and i != 0:
                    print("after " + str(i) + " rounds training, Travel Distance is: " + str(rewards[-1]))

                # Save the variables to disk
                if i % 1000 == 0 and i != 0:
                    save_path = saver.save(sess, config.save_to)
                    print("Model saved in file: %s" % save_path)

            print("Training COMPLETED !")
            save_path = saver.save(sess, config.save_to)
            print("Model saved in file: %s" % save_path)

        # test mode
        else:
            # Get test data
            input_batch = training_set.test_batch()
            feed = {actor.input_: input_batch}

            # Sample solutions
            positions, _, _, _ = sess.run(
                [actor.positions, actor.reward, actor.train_step1, actor.train_step2],
                feed_dict=feed)

            city = input_batch[0]
            position = positions[0]
            result_pos_list = city[position, :]

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率

    if config.training_mode:
        fig = plt.figure()
        plt.plot(list(range(len(rewards))), rewards, c='red')
        plt.title(u"效果曲线")
        plt.xlabel('轮数')
        plt.legend()
        fig.show()
    else:
        fig = plt.figure()
        plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
        plt.title(u"路线")
        plt.legend()
        fig.show()


if __name__ == "__main__":
    main()
