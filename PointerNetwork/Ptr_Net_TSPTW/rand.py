import numpy as np
import time
import random
from Ptr_Net_TSPTW.config import get_config

tasks = []
type = -1

config, _ = get_config()
tasks_num = config.max_length
server_load = config.server_load

alpha = config.alpha
beta = config.beta
gama = config.gama

gen_num = config.nb_epoch


# type 0:随机 1:带权贪心 2:资源贪心 3:优先级贪心 4:超时率贪心
def get_rand_result(tasks_):
    global tasks
    tasks = tasks_
    idx_list = get_idx_list()
    return get_result(idx_list)


def get_idx_list():
    if type == -1:
        return []
    if type == 0:
        return rand_idx_list()
    if type == 1:
        return greed_idx_list()
    if type == 2:
        return greed_1_idx_list()
    if type == 3:
        return greed_2_idx_list()
    if type == 4:
        return greed_3_idx_list()


def rand_idx_list():
    result_idx_list = list(range(len(tasks)))
    random.shuffle(result_idx_list)
    return result_idx_list


def greed_idx_list():
    result_idx_list = []
    task = []
    for task_ in tasks:
        task.append([task_[0] + task_[1] + task_[2] + task_[3], task_[4], task_[5]])
    task = np.array(tasks)

    for i in range(tasks_num):
        rand_idx = random.randint(0, 2)
        min_idx = -1
        if rand_idx == 0:
            min_idx = np.argmin(task[:, 0])
        if rand_idx == 1:
            min_idx = np.argmin(task[:, 1])
        if rand_idx == 2:
            min_idx = np.argmin(task[:, 2])
        task[min_idx, :] = 10000
        result_idx_list.append(min_idx)

    return result_idx_list


def greed_1_idx_list():
    result_idx_list = []
    resource = []
    for task in tasks:
        resource.append(task[0] + task[1] + task[2] + task[3])

    for i in range(tasks_num):
        min_idx = np.argmin(resource)
        resource[int(min_idx)] = 10000
        result_idx_list.append(min_idx)

    return result_idx_list


def greed_2_idx_list():
    result_idx_list = []
    priority = []
    for task in tasks:
        priority.append(task[4])

    for i in range(tasks_num):
        min_idx = np.argmin(priority)
        priority[int(min_idx)] = 10000
        result_idx_list.append(min_idx)

    return result_idx_list


def greed_3_idx_list():
    result_idx_list = []
    timeout = []
    for task in tasks:
        timeout.append(task[5])

    for i in range(tasks_num):
        min_idx = np.argmin(timeout)
        timeout[int(min_idx)] = 10000
        result_idx_list.append(min_idx)

    return result_idx_list


def get_result(result_idx_list):
    global tasks

    task_priority_max = 0
    for i in range(tasks_num):
        task_priority_max = max(task_priority_max, tasks[i][4])

    task_priority_sum = 0
    for idx in range(tasks_num):
        i = result_idx_list[idx]
        task_priority = tasks[i][4]
        task_priority = (task_priority / task_priority_max) * (1 - idx / tasks_num)
        task_priority_sum += task_priority

    ns_ = 0
    time_use = 0
    punish = 0
    server_run_map = []
    server_remain = [1, 1, 1, 1]
    for idx in result_idx_list:
        task = tasks[idx]
        need = task[:4]
        time_out = task[5]
        time_need = task[6]

        if time_use + time_need > time_out:  # 超时
            ns_ += 1
            punish += time_need / server_load
            continue

        while server_remain[0] < need[0] or server_remain[1] < need[1] or \
                server_remain[2] < need[2] or server_remain[3] < need[3]:
            server_run_map = np.array(server_run_map)
            time_use += 1  # 更新时间
            server_run_map[:, -1] -= 1
            server_run_map = server_run_map.tolist()

            while len(server_run_map) > 0:  # 移除已完成的任务
                min_task_idx = np.argmin(server_run_map, axis=0)[-1]
                min_task = server_run_map[min_task_idx]
                min_need = min_task[:4]
                min_time = min_task[-1]
                if min_time > 0:
                    break
                server_remain = np.add(server_remain, min_need)  # 更新剩余容量
                del server_run_map[min_task_idx]  # 移除任务

        server_run_map.append(task)  # 将新任务加入服务器
        server_remain = np.subtract(server_remain, need)  # 更新服务器剩余容量

    max_time_idx = np.argmax(server_run_map, axis=0)[-1]
    max_time = server_run_map[max_time_idx][-1]
    time_use += max_time + punish
    time_use = time_use / tasks_num
    task_priority_sum = task_priority_sum / tasks_num
    ns_prob = ns_ / tasks_num

    result = time_use + task_priority_sum + ns_prob
    return result, time_use, task_priority_sum, ns_prob


def do_rand(input_batch, type_):
    global type
    type = type_
    result_batch = []
    time_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for tasks in input_batch:
        time_start = time.time()
        result, time_result, task_priority_result, ns_result = get_rand_result(tasks)
        time_end = time.time()
        print("rand: ", time_end - time_start)
        result_batch.append(result)
        time_result_batch.append(time_result)
        task_priority_result_batch.append(task_priority_result)
        ns_result_batch.append(ns_result)

    result_array = np.array(result_batch)
    time_result_array = np.array(time_result_batch)
    task_priority_result_array = np.array(task_priority_result_batch)
    ns_result_array = np.array(ns_result_batch)

    result = np.mean(result_array)
    time_result = np.mean(time_result_array, axis=0)
    task_priority_result = np.mean(task_priority_result_array)
    ns_result = np.mean(ns_result_array)

    result = [result] * gen_num
    time_result = [time_result] * gen_num
    task_priority_result = [task_priority_result] * gen_num
    ns_result = [ns_result] * gen_num

    return result, time_result, task_priority_result, ns_result
