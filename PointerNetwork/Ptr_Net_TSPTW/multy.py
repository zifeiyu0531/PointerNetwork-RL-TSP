import numpy as np
from Ptr_Net_TSPTW.config import get_config

config, _ = get_config()

gen_num = config.nb_epoch
tasks_num = config.max_length
server_load = config.server_load
tasks = []
servers = []
raw_tasks = []


def get_multy_result(tasks_):
    global tasks
    tasks = tasks_

    raw_tasks = []
    multy_run_map = [[] for _ in range(5)]
    time_slice = [5, 4, 3, 2, 2]
    for task in tasks:
        multy_run_map[int(task[4])].append(task)
    time_used = 0
    punish = 0
    ns = 0

    for i in range(5):
        server_ = [1, 1, 1, 1]
        temp_run_map = multy_run_map[i]
        if i == 4:  # 最后的队列
            time_used, ns, raw_tasks = do_last_multy(temp_run_map, time_used, ns, server_, raw_tasks, punish)
            break

        for task in temp_run_map:
            if time_used + task[-1] > task[-2]:
                ns += 1
                punish += task[-1] / server_load
                raw_tasks.append(task)
                continue
            if server_[0] < task[0] or server_[1] < task[1] \
                    or server_[2] < task[2] or server_[3] < task[3]:
                time_used += time_slice[i]  # 一次并行完成，总时间增加
                server_ = [1, 1, 1, 1]  # 归还资源

            server_ = np.subtract(server_, task[:4])  # 减去资源
            task[-1] -= time_slice[i]  # 减去时间片
            if task[-1] <= 0:  # 执行完毕
                raw_tasks.append(task)
            else:
                multy_run_map[i + 1].append(task)

    task_priority_max = 0
    for i in range(tasks_num):
        task_priority_max = max(task_priority_max, raw_tasks[i][4])

    task_priority_sum = 0
    for i in range(tasks_num):
        task_priority = raw_tasks[i][4]
        task_priority = (task_priority / task_priority_max) * (1 - i / tasks_num)
        task_priority_sum += task_priority

    time_use = time_used / tasks_num
    task_priority_sum = task_priority_sum / tasks_num
    ns_prob = ns / tasks_num
    result = time_use + task_priority_sum + ns_prob
    return result, time_use, task_priority_sum, ns_prob


def do_last_multy(temp_run_map, time_use, ns_, server_remain, raw_tasks, punish):
    raw_tasks.extend(temp_run_map)
    server_run_map = []
    for task in temp_run_map:
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
    return time_use, ns_, raw_tasks


def do_multy(input_batch):
    result_batch = []
    time_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for tasks in input_batch:
        result, time_result, task_priority_result, ns_result = get_multy_result(tasks)
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
