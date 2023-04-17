# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import matplotlib
import sys
from collections import defaultdict

_config_log = None
_config_out = None

# 这个是用来读mini-batch输出的
def get_time_minibatch(fileName):
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
    
    # print(data)

    # 丢弃mini-batch数量
    data = data[4:10]
    
    f.close()
    
    return np.mean(data)

# 这个是用来读micro-batch输出的
def get_time_microbatch(fileName, mp, is_the_last_micro_batch):
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

            # 丢弃mini-batch数量
            if mini_batch < 4:
                continue
            
            micro_batch_first = line.find(s_micro_batch_begin) + len(s_micro_batch_begin)
            micro_batch_first_end = line.find(s_micro_batch_end)
            micro_batch = int(line[micro_batch_first:micro_batch_first_end])

            time_first = line.find(s_time_begin) + len(s_time_begin)
            time_end = line.find(s_time_end) - 1 #s
            time = float(line[time_first:time_end])

            # print(mini_batch)
            # print(micro_batch)
            # print(time)
            
            if is_the_last_micro_batch:
                if (micro_batch % mp) == mp - 1:
                    data.append(time)
            else:
                data.append(time)
    
    # print(data)
                
    f.close()
    
    return np.mean(data)

def preprocess(model_path, model, dataset, floatType, mp, bs):
    ddpLogFilePath = os.path.join(model_path, "{}/{}/{}_{}_{}_mp{}_bs{}".format('V100', 'DDP_standard', model, dataset, floatType, mp, bs))
    eddpcpuLogFilePath = os.path.join(model_path, "{}/{}/{}_{}_{}_mp{}_bs{}".format('V100', _config_log, model, dataset, floatType, mp, bs))

    print(ddpLogFilePath)
    print(eddpcpuLogFilePath)

    ddpLogFileName = os.path.join(ddpLogFilePath, "mp{}_pod{}".format(mp, 8))
    eddpcpuLogFileName = os.path.join(eddpcpuLogFilePath, "mp{}_pod{}".format(mp, 1))

    ddpTime = get_time_minibatch(ddpLogFileName)
    eddpcpuTime = get_time_microbatch(eddpcpuLogFileName, mp, is_the_last_micro_batch=False)
    eddpcpuTime_thelast = get_time_microbatch(eddpcpuLogFileName, mp, is_the_last_micro_batch=True)

    print(ddpTime)
    print(eddpcpuTime)
    print(eddpcpuTime_thelast)

    return {'ddp': ddpTime, 'eddpcpu': eddpcpuTime, 'eddpcpu_thelast': eddpcpuTime_thelast}


def to_csv(model_list, ddp_ret_dict, eddpcpu_ret_dict, eddpcpu_thelast_ret_dict):
    with open(_config_out,'w') as f:
        f_csv = csv.writer(f)
        header = ['model', 'ddp', 'eddpcpu', 'eddpcpu_thelast']
        f_csv.writerow(header)
        for model in model_list:
            row = [model, ddp_ret_dict[model], eddpcpu_ret_dict[model], eddpcpu_thelast_ret_dict[model]]
            f_csv.writerow(row)
    f.close()


if __name__ == '__main__':
    #!!!! 需要修改目录看这里
    log_path = "../log"
    dataset_dict = {'shufflenetv2': 'imagenet', 'resnet50': 'imagenet', 'vgg19': 'imagenet', 'ncf': 'movielens', 'yolov3': 'voc', 'bert': 'squad', 'electra': 'squad', 'swintransformer': 'imagenet'}
    floatType = 'float32'
    mp = 8
    bs_dict = {'shufflenetv2': 256, 'resnet50': 128, 'vgg19': 64, 'ncf': 16384, 'yolov3': 8, 'bert': 16, 'electra': 16, 'swintransformer': 64}
    
    # 一次性打所有表，生成csv
    model_list = ['shufflenetv2', 'resnet50', 'vgg19', 'ncf', 'yolov3', 'bert', 'electra', 'swintransformer']
    #model_list = ['vgg19', 'yolov3', 'bert', 'swintransformer']

    def output():
        ddp_ret_dict = defaultdict(float)
        eddpcpu_ret_dict = defaultdict(float)
        eddpcpu_thelast_ret_dict = defaultdict(float)
        for model in model_list:
            model_path = log_path + '/' + model
            ret_dict = preprocess(model_path, model, dataset_dict[model], floatType, mp, bs_dict[model])
            ddp_ret_dict[model] = (ret_dict['ddp'])
            eddpcpu_ret_dict[model] = (ret_dict['eddpcpu'])
            eddpcpu_thelast_ret_dict[model] = (ret_dict['eddpcpu_thelast'])

        to_csv(model_list, ddp_ret_dict, eddpcpu_ret_dict, eddpcpu_thelast_ret_dict)

    _config_log = "EDDP_CPU_standard"
    _config_out = "gradient_worker.csv"

    output()
