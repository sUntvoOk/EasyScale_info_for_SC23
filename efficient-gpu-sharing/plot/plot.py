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

time_data = dict()
mem_data = dict()

with open("time.csv", "r") as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        key = row[0]
        val = row[1:]
        assert key not in time_data
        time_data[key] = val

with open("mem.csv", "r") as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        key = row[0]
        val = row[1:]
        assert key not in mem_data
        mem_data[key] = val

def plot(name, label):
    #开始画图
    # Plot configuration
    #height = 0.8
    #fig = plt.figure(frameon=False)
    #plt.style.use('classic')
    plt.rcParams['figure.figsize'] = (4.0, 3) 
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    fs = 15 
    #appcs = ['#010659','#FA2294','#FDD931']
    appcs = ['#e41a1c',
            '#377eb8',
            '#4daf4a']
    appcs = ['#1b9e77',
            '#d95f02',
            '#7570b3']
    #ax.set_title('Performance of swCHOLBLAS')
    
    width = 0.27  # the width of the bars
    space = 0.44
    
    bars = 2
    bars2 = bars*2
    
    mp_label = [int(i) for i in time_data['mp']]
    ind = np.arange(len(mp_label))  # the x locations for the groups
    print (ind)
    
    baseline = float(time_data['{}_ddp'.format(name)][0])
    ddp_time = [baseline/float(t) for t in time_data['{}_ddp'.format(name)] ]
    eddp_time = [baseline/float(t) for t in time_data['{}_eddp'.format(name)] ]

    speedup = [a/b for a,b in zip(ddp_time, eddp_time)]
    print("Speedup {}".format(speedup))
    
    ax.bar(ind-space+space*2/bars2*1, ddp_time , width, label='Worker packing', edgecolor='Black', linewidth=0.4, color='#C0C0C0', hatch='//')
    ax.bar(ind-space+space*2/bars2*3, eddp_time, width, label='EasyScale', edgecolor='Black', linewidth=0.4, color='#FFDEAD', hatch='')
    
    #ax.plot(ind, [1 for i in mtxName], linewidth=0.5, color='black',linestyle=':')
    #ax.plot(ind, [2 for i in mtxName], linewidth=0.5, color='black',linestyle=':')
    #ax.plot(ind, [-1 for i in mtxName], linewidth=0.5, color='black',linestyle=':')
    
    ax.set_xlim(-0.6, len(mp_label)-0.4)
    ax.set_xticks(ind)
    ax.set_xticklabels(mp_label, fontsize=fs-1)
    ax.set_xlabel(label, fontsize=fs)
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    
    ax2 = ax.twinx()
    ddp_mem = [float(t)/1024 for t in mem_data['{}_ddp'.format(name)] ]
    eddp_mem = [float(t)/1024 for t in mem_data['{}_eddp'.format(name)] ]
    
    #ax2.plot([m/len(mp_label) for m in mp_label], ddp_mem , label='DDP', linewidth=0.5, color='#C0C0C0', linestype='--')
    #ax2.plot([m/len(mp_label) for m in mp_label], eddp_mem, label='EasyScale', linewidth=0.5, color='#FFDEAD', linestyle=':')
    ax2.plot(np.array(mp_label)-1, ddp_mem , '-o', label='Worker packing', linewidth=2)
    ax2.plot(np.array(mp_label)-1, eddp_mem, '-^', label='EasyScale', linewidth=2)
    
    oom_pos = 0
    for i, s in enumerate(ddp_time):
        if not s > 0:
            oom_pos = i
            break

    indy = [0, 0.3, 0.6, 0.9, 1.2]
    #while len(indy) > 0:
    #    if indy[-1] > max(ddp_time):
    #        indy = indy[:-1]
    #    else:
    #        break
    ax.set_yticks(indy)
    ax.set_yticklabels([str(i) for i in indy], fontsize=fs)
    #ax.legend(loc='center left', fontsize=fs-2)
    
    indy2 = [0, 8, 16, 24, 32]
    ax2.set_ylim(0, 32)
    ax2.set_yticks(indy2)
    ax2.set_yticklabels([str(i) for i in indy2], fontsize=fs)
    #ax2.legend(loc='center right', fontsize=fs-2)

    if name == "resnet50":
        ax2.tick_params(labelright=False)
        ax.set_ylabel('Normalized Throughput', fontsize=fs)
        #ax.legend(loc='upper center', ncols=2, borderaxespad = 0.)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.2), borderaxespad = 0., ncol=2, fontsize=fs-1, frameon=False, markerscale=1, columnspacing=0.8)
    if name == "shufflenetv2":
        ax.tick_params(labelleft=False)
        ax2.set_ylabel('GPU Memory (GB)', fontsize=fs)
        #ax2.legend(loc='upper center', ncols=2, borderaxespad = 0.)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.55,1.2),borderaxespad = 0., ncol=2, fontsize=fs-1, frameon=False, markerscale=1, columnspacing=0.8)
   
    #plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    fig.savefig('./{}.png'.format(name))
    #plt.show()
    plt.close()


if __name__ == '__main__':
    plot('resnet50', '# of workers')
    plot('shufflenetv2', '# of workers')
