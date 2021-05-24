import random
import numpy as np
from tqdm import tqdm
import time
from Ptr_Net_TSPTW.config import get_config

config, _ = get_config()

chromosome_num = 50
tasks = []
tasks_num = config.max_length
# 迭代轮数
gen_num = config.gen_num

alpha = config.alpha
beta = config.beta
gama = config.gama


def copy_int(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


class Chromosome:
    """
    染色体类
    """

    def __init__(self, genes=None):
        if genes is None:
            genes = [i for i in range(tasks_num)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = 0.0
        self.time_use = 0.0
        self.task_priority = 0.0
        self.ns = 0.0
        self.evaluate_fitness()

    def evaluate_fitness(self):
        task_priority_max = 0
        for i in range(tasks_num):
            task_priority_max = max(task_priority_max, tasks[i][4])

        task_priority_sum = 0
        for idx in range(tasks_num):
            i = self.genes[idx]
            task_priority = tasks[i][4]
            task_priority = (task_priority / task_priority_max) * (1 - idx / tasks_num)
            task_priority_sum += task_priority

        ns_ = 0
        time_use = 0
        server_run_map = []
        server_remain = [1, 1, 1, 1]
        for idx in self.genes:
            task = tasks[idx]
            need = task[:4]
            time_out = task[5]
            time_need = task[6]

            if time_use + time_need > time_out:  # 超时
                ns_ += 1
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
        time_use += max_time
        time_use = time_use / tasks_num
        task_priority_sum = 2 * task_priority_sum / tasks_num
        ns_prob = 2 * ns_ / tasks_num
        self.fitness = time_use + task_priority_sum + ns_prob
        self.time_use = time_use
        self.task_priority = task_priority_sum
        self.ns = ns_prob


class GaAllocate:
    def __init__(self, input):
        self.sumFitness = 0.0
        global tasks
        tasks = input
        self.generation_count = 0
        self.best = Chromosome()
        # 染色体
        self.chromosome_list = []
        # 迭代次数对应的解
        self.result = []
        self.time_result = []
        self.task_priority_result = []
        self.ns_result = []

    @staticmethod
    def cross(parent1, parent2):
        """
        交叉，把第一个抽出一段基因，放到第二段的相应位置
        :param parent1:
        :param parent2:
        :return:
        """
        index1 = random.randint(0, tasks_num - 2)
        index2 = random.randint(index1, tasks_num - 1)
        # crossover at the point cxpoint1 to cxpoint2
        pos1_recorder = {value: idx for idx, value in enumerate(parent1.genes)}
        pos2_recorder = {value: idx for idx, value in enumerate(parent2.genes)}
        for j in range(index1, index2):
            value1, value2 = parent1.genes[j], parent2.genes[j]
            pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
            parent1.genes[j], parent1.genes[pos1] = parent1.genes[pos1], parent1.genes[j]
            parent2.genes[j], parent2.genes[pos2] = parent2.genes[pos2], parent2.genes[j]
            pos1_recorder[value1], pos1_recorder[value2] = pos1, j
            pos2_recorder[value1], pos2_recorder[value2] = j, pos2

        return parent1.genes

    @staticmethod
    def mutate(genes):
        index1 = random.randint(0, tasks_num - 2)
        index2 = random.randint(index1, tasks_num - 1)
        genes_left = genes[:index1]
        genes_mutate = genes[index1:index2]
        genes_right = genes[index2:]
        genes_mutate.reverse()
        return genes_left + genes_mutate + genes_right

    def generate_next_generation(self):
        # 下一代
        for i in range(chromosome_num):
            new_c = self.new_child()
            self.chromosome_list.append(new_c)
            if new_c.fitness < self.best.fitness:
                self.best = new_c
        # chaos
        for i in range(chromosome_num // 2):
            chaos = Chromosome()
            if chaos.fitness < self.best.fitness:
                self.best = chaos
            self.chromosome_list.append(chaos)
        # 锦标赛
        self.chromosome_list = self.champion(self.chromosome_list)

    def new_child(self):
        # 交叉
        parent1 = random.choice(self.chromosome_list)
        parent2 = random.choice(self.chromosome_list)
        new_genes = self.cross(parent1, parent2)
        # 突变
        new_genes = self.mutate(new_genes)
        new_chromosome = Chromosome(new_genes)
        return new_chromosome

    @staticmethod
    def champion(chromosome_list):
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = 5  # 每小组获胜数
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                player = random.choice(chromosome_list)
                player = Chromosome(player.genes)
                group.append(player)
            group = GaAllocate.rank(group)
            winners += group[:group_winner]
        return winners

    @staticmethod
    def rank(chromosome_list):
        for i in range(1, len(chromosome_list)):
            for j in range(0, len(chromosome_list) - i):
                if chromosome_list[j].fitness > chromosome_list[j + 1].fitness:
                    chromosome_list[j], chromosome_list[j + 1] = chromosome_list[j + 1], chromosome_list[j]
        return chromosome_list

    def train(self):
        # 生成初代染色体
        self.chromosome_list = [Chromosome() for _ in range(chromosome_num)]
        self.generation_count = 0
        while self.generation_count < gen_num:
            self.result.append(self.best.fitness)
            self.time_result.append(self.best.time_use)
            self.task_priority_result.append(self.best.task_priority)
            self.ns_result.append(self.best.ns)
            self.generate_next_generation()
            self.generation_count += 1

        return self.result, self.time_result, self.task_priority_result, self.ns_result


def do_ga(input_batch):
    result_batch = []
    time_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for task in tqdm(input_batch):
        time_start = time.time()
        ga = GaAllocate(task)
        result, time_result, task_priority_result, ns_result = ga.train()
        time_end = time.time()
        print("ga: ", time_end - time_start)
        result_batch.append(result)
        time_result_batch.append(time_result)
        task_priority_result_batch.append(task_priority_result)
        ns_result_batch.append(ns_result)

    result_array = np.array(result_batch)
    time_result_array = np.array(time_result_batch)
    task_priority_result_array = np.array(task_priority_result_batch)
    ns_result_array = np.array(ns_result_batch)

    result = np.mean(result_array, axis=0)
    time_result = np.mean(time_result_array, axis=0)
    task_priority_result = np.mean(task_priority_result_array, axis=0)
    ns_result = np.mean(ns_result_array, axis=0)
    return result, time_result, task_priority_result, ns_result
