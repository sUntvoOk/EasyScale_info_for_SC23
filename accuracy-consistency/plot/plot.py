# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib
import json
import os, sys, re
import statistics
import math

matplotlib.get_cachedir()

_stage_batches = 100
stages = ['stage_0_4.txt', 'stage_1_4.txt', 'stage_2_4.txt']
models = ['resnet50', 'vgg19']
#models = ['resnet50', 'shufflenetv2', 'swintransformer', 'vgg19']#, 'ncf', 'yolov3']

def parse_log(log: str, is_ddp: bool) -> dict:
    epoch_start = "epoch: ["
    epoch_end = ", mini-batch: ["
    mini_batch_start = ", mini-batch: ["
    mini_batch_end = ", batch-time:"
    loss_start = ", loss: "
    loss_end = ""

    ret = dict()
    
    with open(log, 'r') as f:
        for line in f:
            if line.find(epoch_start) != -1:
                if not is_ddp and line.find("[MINI_BATCH_LOG]") == -1:
                    continue
                first = line.find(epoch_start) + len(epoch_start)
                end = line.find(epoch_end) - 1
                epoch = int(line[first:end])

                line = line[first:]
                first = line.find(mini_batch_start) + len(mini_batch_start)
                end = line.find(mini_batch_end) - 1
                mini_batch = int(line[first:end])

                line = line[first:]
                first = line.find(loss_start) + len(loss_start)
                string = line[first:]
                if len(string) > 16:
                    string = string[0:15]
                loss = float(string)

                ret[(epoch, mini_batch)] = loss    
    if is_ddp:
        assert len(ret.keys()) >= _stage_batches*len(stages), "LOG file is {}".format(log)
    else:
        assert len(ret.keys()) == _stage_batches, "LOG file is {}".format(log)
    return ret

def get_model_loss(model: str, is_ddp: bool, path="logs"):
    ret = dict()    
    if is_ddp:
        log_file = "../{}/{}/ddp.txt".format(path, model)
        ret["ddp"] = list(parse_log(log_file, is_ddp).values())
    else:
        for stage in stages:
            log_file = "../{}/{}/{}".format(path, model, stage)
            ret[stage] = list(parse_log(log_file, is_ddp).values())
    losses = []
    for key in ret.keys():
        losses += list(ret[key])
    assert len(losses) >= _stage_batches*len(stages)
    return losses

def to_json_ddp(path="logs"):
    data = dict()
    for model in models:
        data["{}_ddp".format(model)] = get_model_loss(path=path, model=model, is_ddp=True)
    
    with open("./{}_database.json".format(path), "w") as f:
        json.dump(data, f)

def to_json_easyscale(path="logs_O0"):
    data = dict()
    for model in models:
        data["{}_easyscale".format(model)] = get_model_loss(path=path, model=model, is_ddp=False)
    
    with open("./{}_database.json".format(path), "w") as f:
        json.dump(data, f)

def replace_name(model):
    model = model.replace("resnet50", "ResNet50")
    model = model.replace("bert", "Bert")
    model = model.replace("vgg19", "VGG19")
    model = model.replace("yolov3", "YOLOv3")
    model = model.replace("swintransformer", "SwinTransformer")
    model = model.replace("electra", "Electra")
    model = model.replace("ncf", "NeuMF")
    model = model.replace("shufflenetv2", "ShuffleNetv2")
    return model


def plot(model, is_hete):
    f = open("logs_homo_database.json", 'r')
    data_homo = json.load(f)
    f.close()
    f = open("logs_hete_database.json", 'r')
    data_hete = json.load(f)
    f.close()
    f = open("logs_O0_database.json", 'r')
    data_O0 = json.load(f)
    f.close()
    f = open("logs_O02_database.json", 'r')
    data_O02 = json.load(f)
    f.close()
    f = open("logs_O1_database.json", 'r')
    data_O1 = json.load(f)
    f.close()
    f = open("logs_O12_database.json", 'r')
    data_O12 = json.load(f)
    f.close()

    # Plot configuration
    #plt.rcParams['figure.figsize'] =(4.0, 2.6)
    plt.rcParams['figure.figsize'] =(6, 4)
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    x = range(1, _stage_batches*len(stages)+1)
    linetype = ['-', '--', ':']
    #linecolor= ['#DDA0DD', '#800080', '#FF0000', '#FA8072']
    linecolor= ['#000000', '#0000FF', '#008000', '#DC1432'] 

    fs = 15

    if not is_hete:
        baseline = data_homo["{}_ddp".format(model)]

        ax.plot(x, [a-b for a,b in zip(data_O0["{}_easyscale".format(model)], baseline)], label="EasyScale-D0")# linestyle=linetype[0], color=linecolor[0])
        ax.plot(x, [a-b for a,b in zip(data_O1["{}_easyscale".format(model)], baseline)], label="EasyScale-D1")# linestyle=linetype[0], color=linecolor[0])
        ax.plot(x, [a-b for a,b in zip(data_homo["{}_ddp".format(model)], baseline)], label="DDP-homo", linestyle="--")#, color=linecolor[0])
    else:
        baseline = data_hete["{}_ddp".format(model)]

        ax.plot(x, [a-b for a,b in zip(data_O02["{}_easyscale".format(model)], baseline)], label="EasyScale-D0+D2")# linestyle=linetype[0], color=linecolor[0])
        ax.plot(x, [a-b for a,b in zip(data_O12["{}_easyscale".format(model)], baseline)], label="EasyScale-D1+D2")# linestyle=linetype[0], color=linecolor[0])
        ax.plot(x, [a-b for a,b in zip(data_hete["{}_ddp".format(model)], baseline)], label="DDP-heter", linestyle="--")#, color=linecolor[0])

    ylim_min, ylim_max = ax.get_ylim()
    xlim_min, xlim_max = ax.get_xlim()
    ax.set_xlim(0, _stage_batches*len(stages))
    y_range = max( math.fabs(ylim_min), math.fabs(ylim_max) )
    if model == "resnet50":
        y_range = 0.26
    ax.set_ylim(-y_range, y_range)

    for idx, stage in enumerate(stages):
        if idx > 0:
            ax.vlines(idx*_stage_batches, -y_range, y_range, color="k", linewidth=0.8)
        plt.text(_stage_batches//2+_stage_batches*idx, -y_range*0.98, "Stage {}".format(idx), horizontalalignment='center', fontsize=fs)

    font = {   'family': 'Times New Roman',
                'weight': 'bold',
                'style':'italic',
                'size': fs, }
    
    plt.text(10, y_range*0.85, replace_name(model), horizontalalignment='left', weight="bold", style="italic", fontsize=fs)

    plt.tick_params(labelsize=fs)
	#labels = ax.get_xticklabels() + ax.get_yticklabels()
	#[label.set_fontname('Nimbus Roman') for label in labels]
	#[label.set_fontstyle('italic') for label in labels]

    ax.set_xticks(range(0, _stage_batches*(len(stages)+1), _stage_batches))
    ax.set_xlabel('Mini-batch', fontsize=fs)
    ax.set_ylabel('Train Loss Difference', fontsize=fs)
    ax.legend(fontsize=fs-2, ncol=1, loc="lower left", bbox_to_anchor=(0,0.08),borderaxespad = 0., frameon=False, labelspacing=0.2)
    plt.tight_layout()
    plt.savefig("{}_{}.png".format(model, 'hete' if is_hete else 'homo'))
    #plt.show()


if __name__ == "__main__":

    to_json_ddp("logs_homo")
    to_json_ddp("logs_hete")
    to_json_easyscale("logs_O0")
    to_json_easyscale("logs_O1")
    to_json_easyscale("logs_O02")
    to_json_easyscale("logs_O12")

    for model in models:
        plot(model, is_hete=False)
        plot(model, is_hete=True)
