#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from tqdm import tqdm

from Ptr_Net_TSPTW.dataset import DataGenerator
from Ptr_Net_TSPTW.actor import Actor
from Ptr_Net_TSPTW.config import get_config, print_config
from Ptr_Net_TSPTW.multy import do_multy
from Ptr_Net_TSPTW.rand import do_rand


# Model: Decoder inputs = Encoder outputs Critic design (state value function approximator) = RNN encoder last hidden
# state (c) (parametric baseline ***) + 1 glimpse over (c) at memory states + 2 FFN layers (ReLu),
# w/o moving_baseline (init_value = 7 for TSPTW20) Penalty: Discrete (counts) with beta = +3 for one constraint /
# beta*sqrt(N) for N constraints violated (concave **0.5) No Regularization Decoder Glimpse = on Attention_g (mask -
# current) Residual connections 01

# NEW data generator (wrap config.py)
# speed1000 model: n20w100
# speed10 model: s10_k5_n20w100 (fine tuned w/ feasible kNN datagen)
# Benchmark: Dumas n20w100 instances


def main():
    # Get running configuration
    config, _ = get_config()
    print_config()

    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config)

    predictions = []
    time_used = []
    task_priority = []
    ns_ = []

    training_set = DataGenerator(config)
    input_batch = training_set.train_batch()

    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # 训练
        if not config.inference_mode:

            # Summary writer
            writer = tf.summary.FileWriter(config.log_dir, sess.graph)

            print("Starting training...")
            for i in tqdm(range(config.nb_epoch)):
                # Get feed dict
                feed = {actor.input_: input_batch}
                # Forward pass & train step

                result, time_use, task_priority_sum, ns_prob, summary, train_step1, train_step2 = sess.run(
                    [actor.reward, actor.time_use, actor.task_priority_sum, actor.ns_prob, actor.merged,
                     actor.train_step1, actor.train_step2],
                    feed_dict=feed)

                time_use = time_use
                ns_prob = ns_prob
                result = time_use + task_priority_sum + ns_prob
                reward_mean = np.mean(result)
                time_mean = np.mean(time_use)
                task_priority_mean = np.mean(task_priority_sum)
                ns_mean = np.mean(ns_prob)

                predictions.append(reward_mean)
                time_used.append(time_mean)
                task_priority.append(task_priority_mean)
                ns_.append(ns_mean)

            print("Training COMPLETED !")

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率

    fig = plt.figure()
    plt.plot(list(range(len(predictions))), predictions, c='red', label=u'指针网络')
    plt.title(u"效果曲线")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(time_used))), time_used, c='red', label=u'指针网络')
    plt.title(u"目标1：运行时间")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(task_priority))), task_priority, c='red', label=u'指针网络')
    plt.title(u"目标2：任务优先级")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(ns_))), ns_, c='red', label=u'指针网络')
    plt.title(u"目标3：超时率")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    rand_result, rand_time_result, rand_task_priority_result, rand_ns_result = do_rand(input_batch, 0)
    greed_result, greed_1_result, greed_2_result, greed_3_result = do_rand(input_batch, 1)
    multy_result, multy_1_result, multy_2_result, multy_3_result = do_multy(input_batch)

    print('task:', config.max_length)
    print('gen_num:', config.gen_num)
    print('nb_epoch:', config.nb_epoch)
    print('ptr')
    print('综合效果', mean(predictions[-10:]))
    print('目标1：运行时间', mean(time_used[-10:]))
    print('目标2：任务优先级', mean(task_priority[-10:]))
    print('目标3：超时率', mean(ns_[-10:]))
    print('greed')
    print('综合效果', mean(greed_result[-10:]))
    print('目标1：运行时间', mean(greed_1_result[-10:]))
    print('目标2：任务优先级', mean(greed_2_result[-10:]))
    print('目标3：超时率', mean(greed_3_result[-10:]))
    print('rand')
    print('综合效果', mean(rand_result[-10:]))
    print('目标1：运行时间', mean(rand_time_result[-10:]))
    print('目标2：任务优先级', mean(rand_task_priority_result[-10:]))
    print('目标3：超时率', mean(rand_ns_result[-10:]))
    print('multy')
    print('综合效果', np.mean(multy_result[-10:]))
    print('目标1：运行时间', np.mean(multy_1_result[-10:]))
    print('目标2：任务优先级', np.mean(multy_2_result[-10:]))
    print('目标3：超时率', np.mean(multy_3_result[-10:]))


if __name__ == "__main__":
    main()
