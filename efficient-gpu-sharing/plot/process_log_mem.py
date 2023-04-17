# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import matplotlib
import sys
from collections import defaultdict


def get_mem(fileName, mp):
    f = open(fileName, 'r')
    lines=f.readlines()
    lines = [line[1:-2] for line in lines]
    rows=[]
    gpus=int(mp)
    max_mem=[float(0) for i in range(gpus)]
    for line in lines:
        rows.append(line.split(','))
    for col in rows:
        for i in range(gpus):
            max_mem[i] = max(max_mem[i], float(col[i]))
    f.close()
    
    return np.sum(max_mem)

def plot(path, model, dataset, floatType, mp, bs):
    ddpLogFilePath = os.path.join(path, "{}/{}_{}_{}_bs{}".format('DDP', model, dataset, floatType, bs))
    eddpcpuLogFilePath = os.path.join(path, "{}/{}_{}_{}_mp{}_bs{}".format('EDDP_CPU', model, dataset, floatType, mp, bs))

    ddpLogFileName = os.path.join(ddpLogFilePath, "mp{}_mem".format(mp))
    eddpcpuLogFileName = os.path.join(eddpcpuLogFilePath, "mp{}_pod{}_mem".format(mp, 1))

    ddpMem = get_mem(ddpLogFileName, 1)
    eddpcpuMem = get_mem(eddpcpuLogFileName, 1)
    
    # print("mp:{}, #############".format(mp))
    # print(ddpMem)
    # print(eddpcpuMem)

    return {'ddp': ddpMem, 'eddpcpu': eddpcpuMem}

def to_csv(ddp_ret_dict, eddpcpu_ret_dict):
    with open('./mem.csv','w') as f:
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
