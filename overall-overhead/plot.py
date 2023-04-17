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

def plot():
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
    
    width = 0.15  # the width of the bars
    space = 0.44
    
    bars = 3
    bars2 = bars*2
    
    x_label = [i for i in time_data.pop(('model', 'gpu'))]
    x_label = list(time_data.keys())
    x_label = [v[0] for v in x_label]
    x_label2 = []
    for m in x_label:
        if m not in x_label2:
            x_label2.append(m)
    x_label = x_label2
    print(x_label)
    ind = np.arange(len(x_label))  # the x locations for the groups
    print (ind)

    norm_data = dict()
    for model in x_label:
        for gpu in ['v100', 'p100', 't4']:
            key = (model, gpu)
            norm = (float(t)/float(time_data[key][0]) for t in time_data[key])
            ddp, ddp_deter, ddp_hetero, ddp_torch = norm
            assert ddp == 1
            ddp_nv = ddp_torch
            ddp_torch = ddp_hetero
            assert key not in norm_data
            norm_data[key] = (ddp, ddp_deter, ddp_nv, ddp_torch)
    print(norm_data)

    def get_bar(gpu):
        data = [ [] for i in range(4)]
        base = [ [] for i in range(4)]
        actual = [ [] for i in range(4)]
        
        for model in x_label:
            key = (model, gpu)
            ddp, ddp_deter, ddp_nv, ddp_torch = norm_data[key]
            #data[0].append(ddp)
            #data[1].append(ddp_deter-ddp)
            #data[2].append(ddp_nv-ddp_deter)
            #data[3].append(ddp_torch-ddp_nv)
            data[0].append(ddp)
            data[1].append(ddp_deter)
            data[2].append(ddp_nv)
            data[3].append(ddp_torch)
            #base[0].append(0)
            #base[1].append(ddp)
            #base[2].append(ddp_deter)
            #base[3].append(ddp_nv)
            base[0].append(0)
            base[1].append(0)
            base[2].append(0)
            base[3].append(0)

            actual[0].append(ddp)
            actual[1].append(ddp_deter)
            actual[2].append(ddp_nv)
            actual[3].append(ddp_torch)
        return data, base, actual

    diff = 0.05

    for gpu in ['v100', 'p100', 't4']:
        for i in range(1,4):

            tmp = np.mean(get_bar(gpu)[2][i]) - np.mean(get_bar(gpu)[2][i-1])
            #tmp = get_bar(gpu)[2][i]
            print("##{} Speedup avg, {}".format(gpu, tmp))
            

    c3=ax.bar(ind-space+space*2/bars2*1+diff*1, get_bar('v100')[0][3] , width, bottom=get_bar('v100')[1][2], label='EasyScale-D1+D2', edgecolor='Black', hatch='oo', color='#FFFFCC')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    #c2=ax.bar(ind-space+space*2/bars2*1+diff*1, get_bar('v100')[0][2] , width, bottom=get_bar('v100')[1][1], label='+ heterogeneous library', edgecolor='Black', hatch='///', color='#FF9966')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    c1=ax.bar(ind-space+space*2/bars2*1+0,    get_bar('v100')[0][1] , width, bottom=get_bar('v100')[1][0], label='EasyScale-D1', edgecolor='Black', hatch='xxx', color='#99CC33')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    c0=ax.bar(ind-space+space*2/bars2*1-diff*1, get_bar('v100')[0][0] , width-0.02, label='Baseline', edgecolor='Black', hatch='', color='#006699')#, linewidth=0.5, color='#C0C0C0', hatch='//')

    print(c0.patches[0].get_facecolor())

    ax.bar(ind-space+space*2/bars2*3+diff*1, get_bar('p100')[0][3] , width, bottom=get_bar('p100')[1][2],  edgecolor='Black', color=c3.patches[0].get_facecolor(), hatch='oo')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    #ax.bar(ind-space+space*2/bars2*3+diff*1, get_bar('p100')[0][2] , width, bottom=get_bar('p100')[1][1],  edgecolor='Black', color=c2.patches[0].get_facecolor(), hatch='///')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    ax.bar(ind-space+space*2/bars2*3+0   , get_bar('p100')[0][1] , width, bottom=get_bar('p100')[1][0],  edgecolor='Black', color=c1.patches[0].get_facecolor(), hatch='xxx')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    ax.bar(ind-space+space*2/bars2*3-diff*1, get_bar('p100')[0][0] , width-0.02,  edgecolor='Black', color=c0.patches[0].get_facecolor(), hatch='')#, linewidth=0.5, color='#C0C0C0', hatch='//')

    ax.bar(ind-space+space*2/bars2*5+diff*1, get_bar('t4')[0][3] , width, bottom=get_bar('t4')[1][2],  edgecolor='Black', color=c3.patches[0].get_facecolor(), hatch='oo')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    #ax.bar(ind-space+space*2/bars2*5+diff*1, get_bar('t4')[0][2] , width, bottom=get_bar('t4')[1][1],  edgecolor='Black', color=c2.patches[0].get_facecolor(), hatch='xxx')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    ax.bar(ind-space+space*2/bars2*5+0   , get_bar('t4')[0][1] , width, bottom=get_bar('t4')[1][0],  edgecolor='Black', color=c1.patches[0].get_facecolor(), hatch='xxx')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    ax.bar(ind-space+space*2/bars2*5-diff*1, get_bar('t4')[0][0] , width-0.02,  edgecolor='Black', color=c0.patches[0].get_facecolor(), hatch='')#, linewidth=0.5, color='#C0C0C0', hatch='//')
    
    #ax.bar(ind-space+space*2/bars2*3, get_bar('p100'), width, label='EST 0-6', edgecolor='Black', linewidth=0.5, color='#FFDEAD', hatch='')
    #ax.bar(ind-space+space*2/bars2*5, get_bar('t4'), width, label='EST 7', edgecolor='Black', linewidth=0.5, color='#FFDEAD', hatch='-')
    
    ind2 = np.arange(len(x_label)+2)  # the x locations for the groups
    ax.plot(ind2-1, [1 for i in list(ind2)], linewidth=0.5, color='black',linestyle=':')
    ax.plot(ind2-1, [2 for i in list(ind2)], linewidth=0.5, color='black',linestyle=':')
    ax.plot(ind2-1, [3 for i in list(ind2)], linewidth=0.5, color='black',linestyle=':')
    ax.plot(ind2-1, [4 for i in list(ind2)], linewidth=0.5, color='black',linestyle=':')
    ax.plot(ind2-1, [5 for i in list(ind2)], linewidth=0.5, color='black',linestyle=':')

    length=7
    x_label_cut = []
    for name in x_label:
        if len(name) > length:
            name = name[0:(length-1)] + '*'
        x_label_cut.append(name)

    #x_label_cut = ['shufflenet', 'resnet50', 'vgg19', 'yolov3', 'neumf', 'bert', 'electra', 'swin\\transformer']
 
    
    ax.set_xlim(-0.6, len(x_label)-0.4)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_label_cut, fontsize=fs+1)
    
    #for tick in ax.get_xticklabels():
    #    tick.set_rotation(60)
    
    #oom_pos = 0
    #for i, s in enumerate(ddp_time):
    #    if not s > 0:
    #        oom_pos = i
    #        break
    #ax.text(oom_pos, 0.05, "OOM", horizontalalignment='center', fontsize=fs+1)

    indy = [0,1,2,3,4,5]
    #while len(indy) > 0:
    #    if indy[-1] > max(ddp_time):
    #        indy = indy[:-1]
    #    else:
    #        break
    ax.set_yticks(indy)
    ax.set_yticklabels([str(i) for i in indy], fontsize=fs-1)
    
    ax.legend(loc='upper right', fontsize=fs-1)
    ax.set_ylabel('Normalized Time', fontsize=fs)
    
    plt.tight_layout()
    fig.savefig('./overhead.png')
    #plt.show()
    plt.close()


if __name__ == '__main__':
    time_data = dict()
    with open("overhead.csv", "r") as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            model = row[0]
            bs = row[1]
            gpu = row[2]
            key = (model, gpu)
            val = tuple(row[3:])
            assert key not in time_data
            time_data[key] = val
            print(key, val)

    plot()
