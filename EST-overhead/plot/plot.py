# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import re
import math
import pandas as pd
import csv

#mpl.use('Agg')
model_list = ['shufflenetv2', 'resnet50', 'vgg19', 'ncf', 'yolov3', 'bert', 'electra', 'swintransformer']

def replace_name(model_list):
    ret = []
    for model in model_list:
        model = model.replace("resnet50", "ResNet50")
        model = model.replace("shufflenetv2", "ShuffleNetv2")
        model = model.replace("vgg19", "VGG19")
        model = model.replace("ncf", "NeuMF")
        model = model.replace("yolov3", "YOLOv3")
        model = model.replace("bert", "Bert")
        model = model.replace("electra", "Electra")
        model = model.replace("swintransformer", "SwinTransformer")
        ret.append(model)
    return ret

def plot_gradient_worker():
    #开始画图
    # Plot configuration
    #height = 0.8
    #fig = plt.figure(frameon=False)
    #plt.style.use('classic')
    plt.rcParams['figure.figsize'] = (11.0, 4.0) 
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    fs = 22 
    #appcs = ['#010659','#FA2294','#FDD931']
    appcs = ['#e41a1c',
            '#377eb8',
            '#4daf4a']
    appcs = ['#1b9e77',
            '#d95f02',
            '#7570b3']
    #ax.set_title('Performance of swCHOLBLAS')
    
    width = 0.2  # the width of the bars
    space = 0.34
    
    bars = 3
    bars2 = bars*2
    
    
    x_label = [i for i in time_data.pop('model')]
    x_label = list(time_data.keys())
    print(x_label)
    ind = np.arange(len(x_label))  # the x locations for the groups
    print (ind)
    
    ddp_time = [float(v[0])/float(v[0]) for v in list(time_data.values())]
    eddp_first_time = [float(v[1])/float(v[0]) for v in list(time_data.values())]
    eddp_last_time = [float(v[2])/float(v[0]) for v in list(time_data.values())]

    print(ddp_time)

    print("## 0-6 is {}".format(1-np.mean(eddp_first_time)))
    print("## 7 is {}".format(1-np.mean(eddp_last_time)))
    print("## avg is {}".format( 1-(np.mean(eddp_last_time) + np.mean(eddp_first_time)*7)/8 ))

    ax.bar(ind-space+space*2/bars2*1, ddp_time , width, label='DDP 0-7', edgecolor='Black', linewidth=0.5, color='#C0C0C0', hatch='//')
    ax.bar(ind-space+space*2/bars2*3, eddp_first_time, width, label='EST 0-6', edgecolor='Black', linewidth=0.5, color='#FFDEAD', hatch='')
    ax.bar(ind-space+space*2/bars2*5, eddp_last_time, width, label='EST 7', edgecolor='Black', linewidth=0.5, color='#FFDEAD', hatch='--')
    

    ind2 = np.arange(len(x_label)+2)  # the x locations for the groups
    ax.plot(ind2-1, [1 for i in list(ind2)], linewidth=0.5, color='black',linestyle=':')

    length=7
    x_label_cut = []
    for name in replace_name(x_label):
        if len(name) > length:
            name = name[0:(length-1)] + '*'
        x_label_cut.append(name)
 
    ax.set_xlim(-0.6, len(x_label)-0.4)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_label_cut, fontsize=fs)
    
    #for tick in ax.get_xticklabels():
    #    tick.set_rotation(60)
    
    #oom_pos = 0
    #for i, s in enumerate(ddp_time):
    #    if not s > 0:
    #        oom_pos = i
    #        break
    #ax.text(oom_pos, 0.05, "OOM", horizontalalignment='center', fontsize=fs+1)

    indy = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #while len(indy) > 0:
    #    if indy[-1] > max(ddp_time):
    #        indy = indy[:-1]
    #    else:
    #        break
    ax.set_yticks(indy)
    ax.set_yticklabels([str(i) for i in indy], fontsize=fs-1)
    
    ax.legend(loc='lower center', fontsize=fs-2, ncol=3, bbox_to_anchor=(0.5,0.9))
    ax.set_ylabel('Normalized Time', fontsize=fs)
    
    plt.tight_layout()
    fig.savefig('./gradient_worker.png')
    #plt.show()
    plt.close()

def plot_context_switch():
    #开始画图
    # Plot configuration
    #height = 0.8
    #fig = plt.figure(frameon=False)
    #plt.style.use('classic')
    plt.rcParams['figure.figsize'] = (11.0, 4.0) 
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    fs = 22 
    #appcs = ['#010659','#FA2294','#FDD931']
    appcs = ['#e41a1c',
            '#377eb8',
            '#4daf4a']
    appcs = ['#1b9e77',
            '#d95f02',
            '#7570b3']
    #ax.set_title('Performance of swCHOLBLAS')
    
    width = 0.3  # the width of the bars
    space = 0.44
    
    bars = 2
    bars2 = bars*2
    
    
    x_label = [i for i in time_data.pop('model')]
    x_label = list(time_data.keys())
    print(x_label)
    ind = np.arange(len(x_label))  # the x locations for the groups
    print (ind)
    
    eddp_wo = [float(v[0])/float(v[0]) for v in list(time_data.values())]
    eddp_wi = [max(1.0, float(v[1])/float(v[0])) for v in list(time_data.values())]

    print(eddp_wi)
    print(np.mean(eddp_wi))


    ax.bar(ind-space+space*2/bars2*1, eddp_wo , width, label='EasyScale w/o context switching ', edgecolor='Black', linewidth=0.5, color='#FFDEAD', hatch='--')
    ax.bar(ind-space+space*2/bars2*3, eddp_wi, width, label='EasyScale w/ context switching', edgecolor='Black', linewidth=0.5, color='#FFDEAD', hatch='||')
    
    ind2 = np.arange(len(x_label)+2)  # the x locations for the groups
    ax.plot(ind2-1, [1 for i in list(ind2)], linewidth=0.5, color='black',linestyle=':')

    length=7
    x_label_cut = []
    for name in replace_name(x_label):
        if len(name) > length:
            name = name[0:(length-1)] + '*'
        x_label_cut.append(name)
 
    
    ax.set_xlim(-0.6, len(x_label)-0.4)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_label_cut, fontsize=fs)

    ax.set_ylim(0.95, 1.06)
    
    #for tick in ax.get_xticklabels():
    #    tick.set_rotation(60)
    
    #oom_pos = 0
    #for i, s in enumerate(ddp_time):
    #    if not s > 0:
    #        oom_pos = i
    #        break
    #ax.text(oom_pos, 0.05, "OOM", horizontalalignment='center', fontsize=fs+1)

    indy = [0.95, 1.00, 1.05]
    #while len(indy) > 0:
    #    if indy[-1] > max(ddp_time):
    #        indy = indy[:-1]
    #    else:
    #        break
    ax.set_yticks(indy)
    ax.set_yticklabels([str(i) for i in indy], fontsize=fs-1)
    
    ax.legend(loc='upper left', ncol=1, fontsize=fs-2)
    ax.set_ylabel('Normalized Time', fontsize=fs)
    
    plt.tight_layout()
    fig.savefig('./context_switch.png')
    #plt.show()
    plt.close()



if __name__ == '__main__':
    time_data = dict()
    with open("gradient_worker.csv", "r") as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            key = row[0]
            val = row[1:]
            assert key not in time_data
            time_data[key] = val

    plot_gradient_worker()


    time_data = dict()
    with open("context_switch.csv", "r") as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            key = row[0]
            val = row[1:]
            assert key not in time_data
            time_data[key] = val

    plot_context_switch()
