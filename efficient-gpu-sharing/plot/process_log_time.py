# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import matplotlib
import sys
from collections import defaultdict

def get_time_ddp(fileName):
    data = []
    f = open(fileName, 'r')

    s_time = "step-time: "
    s_time_end = "s, loss:"
    # s_time = "batch-time: "
    # s_time_end = ", step-time"
    for line in f:
        if line.find(s_time) != -1:
            first = line.find(s_time) + len(s_time)
            end = line.find(s_time_end) - 1
            time = float(line[first:end])
            data.append(time)
        
    data = data[2:10]

    # print(data)
    
    f.close()
    
    return np.mean(data)

def get_time_eddp(fileName):
    data = []
    f = open(fileName, 'r')

    s_mini_batch_begin = "mini-batch: ["
    s_mini_batch_end = "], micro-batch: "
    s_micro_batch_begin = "micro-batch: ["
    s_micro_batch_end = "], step-time: "
    s_time_begin = "step-time: "
    s_time_end = "s, loss:"
    # s_time_begin = "batch-time: "
    # s_time_end = "s, step-time:"
    for line in f:
        if line.find("MICRO_BATCH_LOG") != -1:
            mini_batch_first = line.find(s_mini_batch_begin) + len(s_mini_batch_begin)
            mini_batch_first_end = line.find(s_mini_batch_end)
            mini_batch = int(line[mini_batch_first:mini_batch_first_end])

            if mini_batch < 2:
                continue
            
            micro_batch_first = line.find(s_micro_batch_begin) + len(s_micro_batch_begin)
            micro_batch_first_end = line.find(s_micro_batch_end)
            micro_batch = int(line[micro_batch_first:micro_batch_first_end])

            time_first = line.find(s_time_begin) + len(s_time_begin)
            time_end = line.find(s_time_end) - 1 #s
            time = float(line[time_first:time_end])
            
            data.append(time)
            
    f.close()
    
    return np.mean(data)

def plot(path, model, dataset, floatType, mp, bs):
    ddpLogFilePath = os.path.join(path, "{}/{}_{}_{}_bs{}".format('DDP', model, dataset, floatType, bs))
    eddpcpuLogFilePath = os.path.join(path, "{}/{}_{}_{}_mp{}_bs{}".format('EDDP_CPU', model, dataset, floatType, mp, bs))

    ddpLogFileName = os.path.join(ddpLogFilePath, "mp{}".format(mp))
    eddpcpuLogFileName = os.path.join(eddpcpuLogFilePath, "mp{}_pod{}".format(mp, 1))

    ddpTime = get_time_ddp(ddpLogFileName)/mp
    eddpcpuTime = get_time_eddp(eddpcpuLogFileName)
    
    # print("mp:{}, #############".format(mp))
    # print(ddpTime)
    # print(eddpcpuTime)

    return {'ddp': ddpTime, 'eddpcpu': eddpcpuTime}

def to_csv(ddp_ret_dict, eddpcpu_ret_dict):
    with open('./time.csv','w') as f:
        f_csv = csv.writer(f)
        header = ['mp'] + list(range(1, 17, 1))
        f_csv.writerow(header)
        for model in ['shufflenetv2', 'resnet50']:
            row_ddp = [model + '_ddp'] + ddp_ret_dict[model]
            f_csv.writerow(row_ddp)
            row_eddp = [model + '_eddp'] + eddpcpu_ret_dict[model]
            f_csv.writerow(row_eddp)
    # print(f_csv)
    f.close()


if __name__ == '__main__':
    path = "../log"
    dataset_dict = {'resnet50': 'imagenet', 'shufflenetv2': 'imagenet'}
    floatType = 'float32'
    bs_dict = {'resnet50': 32, 'shufflenetv2': 512}

    # model='shufflenetv2'
    model='resnet50'
    model_list = ['shufflenetv2', 'resnet50']
    mp_list = range(1, 17, 1)
    ddp_ret_dict = defaultdict(list)
    eddpcpu_ret_dict = defaultdict(list)
    for model in model_list:
        for mp in mp_list:
            ret_dict = plot(path, model, dataset_dict[model], floatType, mp, bs_dict[model])
            ddp_ret_dict[model].append(ret_dict['ddp'])
            eddpcpu_ret_dict[model].append(ret_dict['eddpcpu'])
    
    to_csv(ddp_ret_dict, eddpcpu_ret_dict)
