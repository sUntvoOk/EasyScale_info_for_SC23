import collections
from collections import OrderedDict
import functools
import glob
import math
import os, sys
import pandas
import numpy as np
import copy
import json

import pdb

from logger import my_logger
logger = my_logger

from throughput import ThroughputFunction, fit_perf_params
from easy_policy import Intrajob

from scipy.interpolate import interp1d, LinearNDInterpolator

def get(name):
    return APPLICATIONS[name]

def memoize(f):
    memo = {}
    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]
    return helper



class Application(object):
    def __init__(self, trace_dir, ckpt_time, epochs, dataset_len):
        self.name = os.path.basename(trace_dir)
        self.ckpt_time = ckpt_time
        self.epochs = epochs
        self.dataset_len = dataset_len
        self.gpu_types = ('v100', 'p100', 't4')
        self.gpu_flops=(15.7, 10.6, 8.1)

        self.logs = None
        with open(os.path.join(trace_dir, 'logs.json'), 'r') as f:
            self.logs = json.load(f)

        # scalability.csv 根据nodes和pods对性能进行拟合，这个同样可以适配异构场景
        #self.placements = \
        #    pandas.read_csv(os.path.join(trace_dir, "placements.csv"))
        #for gpu in self.gpu_types:
        #    self.placements[gpu+"_num_nodes"] = \
        #        getattr(self.placements, gpu+'_placement').apply(lambda p: len(str(p)) if int(p) > 0 else 0  )
        #    # 目前只考虑GPU的个数，暂时不考虑放置情况。
        #    # 有一个基本的假设：在不使用nvlink的情况下，所有pod之间都走以太网，放置情况对扩展性的影响不大。
        #    self.placements[gpu+"_num_pods"] = \
        #        getattr(self.placements, gpu+'_placement').apply(lambda p: sum(map(int, str(p))))
            
        self.proposals = None

        self.gpu_perf = dict()
        self.profile = dict()
        self.perf_params = None

        self._preprocess_the_logs()
    
    def _preprocess_the_logs(self):
        logs = self.logs

        self.gpu_perf = {gpu:dict() for gpu in self.gpu_types}
        for batch_size in logs:
            for i, pr in enumerate(logs[batch_size]['step1']):
                #logger.info("## bs={}, pr is {}".format(batch_size, pr))
                key = self.gpu_types[i]
                val = pr['total_time']
                self.gpu_perf[key].update({batch_size:val})
        print(self.gpu_perf)
        
        # for batch_size in logs:
        #     time = list(self.get_gpu_time(batch_size).values())
        #     perf = [ time[-1]/i for i in time ]
        #     baseline = list(self.get_gpu_time(batch_size).values())[-1]
        #     proposals = logs[batch_size].get('proposals', None)
        #     if not proposals:
        #         continue
        #     for pr in proposals:
        #         placement = list(pr['placement'].values())
        #         thread = list(pr['threads'].values())
        #         for i in range(len(placement)):
        #             if placement[i] == 0:
        #                 thread[i] = 0
        #         placement = tuple(placement)
        #         thread = tuple(thread)

        #         #comp_time = baseline / pr['speedup'] * pr['mp']
        #         ## TODO: correct speedup
        #         speedup = sum( [a*b for a,b in zip(perf, placement)] ) * (1 - pr['waste'])
        #         #logger.info("@@ {}, {}, correct speedup is {}".format(list(perf), list(placement), speedup))
        #         comp_time = baseline * pr['mp'] / speedup
        #         #comp_time = max([a*b for a,b in zip(perf, thread)])
        #         total_time = pr['total_time']

        #         # filter the profiles

        #         #if thread in [(1,0,0), (0,1,0), (0,0,1)]:
        #         #    continue
        #         if thread[1] > 0 or thread[2] > 0:
        #             continue 
        #         if placement[0] % 3 == 0:
        #             continue
        #         #if placement[2] % 3 == 0 or placement[0] % 3 == 0 or placement[1] % 3 == 0:
        #         #    continue
        #         if total_time is None:
        #             continue
        #         if total_time < comp_time:
        #             continue

        #         if (total_time is not None) and (total_time is not np.nan) and (total_time > 0):
        #             key = (placement, thread, comp_time)
        #             val = total_time
        #             if key in self.profile:
        #                 assert "Maybe exist some duplication in logs"
        #             self.profile[key] = val
        #             #print("@@ key={}, val={}".format(key,val))
        
        # ## TODO:: 要设计一些插值方法，能预测任何配置下的
        # # 目前先用我建立的性能模型替代
        # placements = [key[0] for key in self.profile]
        # threads = [key[1] for key in self.profile]
        # comp_times = [key[2] for key in self.profile]
        # total_times = [val for val in self.profile.values()]
        # self.perf_params = fit_perf_params(placements, threads, comp_times, total_times)

        # throughput_fn = ThroughputFunction(self.perf_params)

        # for k,v in self.profile.items():
        #     placement = {a:b for a,b in zip(self.gpu_types, k[0])}
        #     thread = {a:b for a,b in zip(self.gpu_types, k[1])}
        #     comp_time = k[2]
        #     
        #     t = throughput_fn.get_time(placement, thread, comp_time)
        #     err = (t-v)/v * 100
        #     if np.abs(err) > 10:
        #         print(k, v, t, '\t\t\t\t', err)

        # #print(placements[0], threads[0], comp_times[0], total_times[0])

    
    def get_step_time(self, placement, thread, comp_time):
        throughput_fn = ThroughputFunction(self.per_params)
        return throughput_fn(placement, thread, comp_time)

    #def get_step_time(self, proposal, batch_size):
    #    # time to finish a global mini batch
    #    ## TODO : 目前用intrajob的waste函数近似
    #    df = self.placements[  (self.placements['v100_placement'] == 0) & 
    #                            (self.placements['p100_placement'] == 0) & 
    #                            (self.placements['t4_placement'] == 1) ] 
    #    interpolator= interp1d(df.local_bsz, df.step_time, axis=0)

    #    baseline = interpolator(batch_size)
    #    step_time = baseline / proposal['speedup']
    #    return step_time

    def get_comp_time(self, threads, batch_size):
        # time to finish a global mini batch without allreduce
        gpu_time = self.get_gpu_time(batch_size).values()
        threads = threads.values()
        time = [a*b for a,b in zip(gpu_time, threads)]
        return max(time)


    def get_gpu_time(self, batch_size):
        # time to train one mini-batch

        interpolators = dict()
        for gpu in self.gpu_types:
            interpolators[gpu] = interp1d([int(i) for i in self.gpu_perf[gpu].keys()], list(self.gpu_perf[gpu].values()), axis=0, fill_value="extrapolate")
        
        perf = dict()
        for gpu in self.gpu_types:
            perf[gpu] = float(interpolators[gpu](batch_size))
        values = list(perf.values())
        
        return dict(zip(self.gpu_types, values))
    
    # @memoize
    def get_proposals(self, max_parallelism, batch_size, gpu_nums=None):
        # return a list of placements
        gpu_time = list( self.get_gpu_time(batch_size).values() )
        gpu_perf = [ gpu_time[-1]/i for i in gpu_time ]

        #logger.info("@@ perf is {}".format(gpu_perf))
        intra = Intrajob(   max_parallelism=max_parallelism,
                            gpu_types=self.gpu_types,
                            gpu_flops=self.gpu_flops,
                            gpu_perf=gpu_perf,
                            gpu_nums=gpu_nums   )
        return intra.get_proposals()


TRACES_DIR = os.path.join(os.path.dirname(__file__), "../logs/")
APPLICATIONS = {
    "resnet50": Application(os.path.join(TRACES_DIR, "resnet50"), ckpt_time=20, epochs=1, dataset_len=1280000*0.01), 
    "vgg19": Application(os.path.join(TRACES_DIR, "vgg19"), ckpt_time=20, epochs=1, dataset_len=1280000*0.01), 
    "shufflenetv2": Application(os.path.join(TRACES_DIR, "shufflenetv2"), ckpt_time=20, epochs=1, dataset_len=1280000*0.01), 
    "swintransformer": Application(os.path.join(TRACES_DIR, "swintransformer"), ckpt_time=20, epochs=1, dataset_len=1280000*0.01), 
    "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), ckpt_time=20, epochs=1, dataset_len=16551*0.01), 
    "bert": Application(os.path.join(TRACES_DIR, "bert"), ckpt_time=20, epochs=1, dataset_len=88641*0.01), 
    "electra": Application(os.path.join(TRACES_DIR, "electra"), ckpt_time=20, epochs=1, dataset_len=88641*0.01), 
    "ncf": Application(os.path.join(TRACES_DIR, "ncf"), ckpt_time=20, epochs=1, dataset_len=4970845*0.01), 
    #"vgg19": Application(os.path.join(TRACES_DIR, "vgg19"), ckpt_time=20, epochs=20, dataset_len=1280000, heterogeneous_deterministic=False),
    #"shufflenetv2": Application(os.path.join(TRACES_DIR, "shufflenetv2"), ckpt_time=20, epochs=20, dataset_len=1280000, heterogeneous_deterministic=False)
}


if __name__ == '__main__':
    #intra = Intrajob(max_parallelism=16, gpu_types=['v100', 'p100', 't4'], gpu_flops=[15.7, 10.6, 8.1], gpu_perf=[3.1,2.4,1], gpu_nums=[0,0,16])
    #intra = Intrajob(max_parallelism=16, gpu_names=['v100', 'p100', 't4'], gpu_flops=[15.7, 10.6, 8.1], gpu_perf=[3.1,2.4,1], gpu_nums=[32,0,0])
    #intra = Intrajob(max_parallelism=16, gpu_names=['v100'], gpu_flops=[15.7], gpu_perf=[3.1], gpu_nums=[32])

    #intra = Intrajob(max_parallelism=16, gpu_types=['v100', 'p100', 't4'], gpu_flops=[15.7, 10.6, 8.1], gpu_perf=[3.1,2.4,1], gpu_nums=[32,16,16])
    #intra.get_proposals()

    app = get("resnet50")
    #app = get("cifar10-vgg19")
    print(app.get_gpu_time(32))
    print(app.get_gpu_time(64))
    print(app.get_gpu_time(128))
    print(app.get_gpu_time(256))

    #print(app.perf_params)

    #app.get_proposals(max_parallelism=16, gpu_nums=[32,16,16])
    #app.get_proposals(16, (32,16,16))
    #t = app.get_step_time(app.get_proposals(16, (32,16,16))[0], 32)
    #print(t)
    app.get_proposals(16, 128, (0,0,16))
