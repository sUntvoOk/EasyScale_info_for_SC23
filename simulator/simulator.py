import argparse
import collections
from collections import OrderedDict
import copy
import glob
import json
import math
import multiprocessing
import os

import numpy as np
import pandas
import pdb

from applications import APPLICATIONS
from easy_policy import EasyPolicy
from other_policy import FifoPolicy, SrtfPolicy
from throughput import ThroughputFunction, fit_perf_params

from logger import my_logger
logger = my_logger


class Job(object):

    def __init__(self, name, application, submission_time,
                 max_parallelism, batch_size, heterogeneous_deterministic, gpu_nums, epochs, preferred_gpu):
        self.name = name
        self.application = application
        self.submission_time = submission_time
        self.max_parallelism = max_parallelism
        self.batch_size = batch_size
        self.heterogeneous_deterministic = heterogeneous_deterministic
        #self.min_parallelism = min_parallelism
        self.gpu_nums = gpu_nums
        self.epochs = epochs if epochs else application.epochs
        self.preferred_gpu = preferred_gpu
        self.dataset_len = application.dataset_len
        self.total_progress = self.epochs * self.dataset_len 
        self.total_batch_size = self.batch_size * self.max_parallelism

        self.completion_time = None
        self.current_time = 0
        self.rescale_time = 20  # Start re-scale countdown.
        # Total GPU number, 
        # e.g., (1,1,1) --> 1 * V100, 1 * P100, 1 * T4
        # e.g., {"V100":2, "P100":1, "T4":1}
        # GPU locations, 
        # e.g., (1,0,0,0,1,1) --> the GPU locations on six nodes
        # e.g., {"V100":[1,1], "P100":[1], "T4":[1]}
        # Note that the nodes are equipped with different types of GPU.
        # ref: Node.type
        self.placement = ()
        self.proposals = None
        # accumulated profiling results
        self.profile = {}
        self.perf_params = None
        self.epoch = 0
        self.num_restarts = None
        self.progress = 0

        self._max_waste = 0.30

        ## TODO: hack the perf_params with the application's
        self.perf_params = copy.deepcopy(self.application.perf_params)

    def _fill_the_profile(self):
        logs = self.application.logs
        for batch_size in logs:
            baseline = self.get_gpu_time(batch_size).values()[-1]
            proposals = logs[batch_size].get('proposals', None)
            if not proposals:
                continue
            for pr in proposals:
                placement = tuple(pr['placement'].values())
                threads = tuple(pr['threads'])
                comp_time = baseline / pr['speedup']
                total_time = pr['total_time']
                key = (placement, threads, comp_time)
                if key in self.profile:
                    assert "Maybe exist some duplication in logs"
                self.profile[key] = total_time

    def get_gpu_time(self):
        perf = self.application.get_gpu_time(self.batch_size)
        return perf

    def get_throughput_fn(self):
        return ThroughputFunction(self.perf_params)

    # def pred_throughput(self, proposal):
    #     # return the throughput under this placement
    #     # e.g., 1000 images per second
    #     comp_time = self.application.get_comp_time(proposal['threads'], self.batch_size)
    #     total_batch_size = self.total_batch_size

    #     throughput_fn = self.get_throughput_fn()
    #     tp = throughput_fn(proposal['placement'], proposal['threads'], comp_time, total_batch_size)
    #     t = self.batch_size * self.max_parallelism / tp
    #     return tp

    def pred_throughput(self, proposal):
        # return the throughput under this placement
        # e.g., 1000 images per second
        comp_time = self.application.get_comp_time(proposal['threads'], self.batch_size)
        total_batch_size = self.total_batch_size

        tp = self.batch_size * self.max_parallelism / comp_time
        return tp

    def get_placement_proposals(self, force_homo=False, force_non_elastic=False):
        if not self.proposals:
            self.proposals = self.application.get_proposals(self.max_parallelism, self.batch_size, list(self.gpu_nums.values()))

            #for pr in self.proposals:
            #    pr.update({'total_time': self.total_batch_size/self.pred_throughput(pr)})

            #for p in self.proposals:
            #    logger.info("@@ proposal {}".format(p))
        proposals_filtered = []


        if (not self.heterogeneous_deterministic) or force_homo  or force_non_elastic:
            for p in self.proposals:
                #if p['placement']['p100'] == 0 and p['placement']['t4'] == 0:
                flag = {gpu:False for gpu in self.application.gpu_types}
                if not force_non_elastic:
                    flag[self.preferred_gpu] = True
                else:
                    if p['placement'][self.preferred_gpu] == self.max_parallelism:
                        flag[self.preferred_gpu] = True

                for gpu in self.application.gpu_types:
                    if gpu is not self.preferred_gpu:
                        if p['placement'][gpu] == 0:
                            flag[gpu] = True
                if all(f is True for f in flag.values()):
                    proposals_filtered.append(p)
            self.proposals = proposals_filtered 

        assert len(self.proposals) > 0

        return self.proposals

    def update_progress(self, throughput, time):
        self.progress += throughput * time

    def update_params(self, placement, thread, comp_time, total_time):
        ## TODO: hack the perf_params with the application's
        pass
        #if (tuple(placement), tuple(thread), comp_time) in self.profile:
        #    return
        #self.profile[(tuple(placement), tuple(thread), comp_time)] = total_time

        #placements = np.array([key[0] for key in self.profile])
        #threads = np.array([key[1] for key in self.profile])
        #comp_times = np.array([key[2] for key in self.profile])
        #total_times = np.array([val for val in self.profile.values()])

        #self.perf_params = fit_perf_params(placements, threads, comp_times, total_times)

    def submit_proposals(self):
        ## 根据当前的性能模型参数更新speedup的数值
        #for pr in self.proposals:
        #    pr.update({'total_time': self.total_batch_size/self.pred_throughput(pr)})

        available_proposals = {gpu:[] for gpu in self.application.gpu_types}

        proposals = copy.deepcopy(self.get_placement_proposals())

        def judge(gpu, p, cur_p):
            flag = {g: False for g in self.application.gpu_types}
            for g in self.application.gpu_types:
                if g == gpu:
                    if (p['placement'][gpu] > cur_p['placement'][gpu]) :
                        flag[gpu] = True
                else:
                    if (p['placement'][g] <= cur_p['placement'][g]) :
                        flag[g] = True
            return all(f is True for f in flag.values())


        # 重排proposal，按照speedup的降序排列
        proposals = sorted(proposals, key=lambda x:x['speedup'], reverse=True)

        cur_p = copy.deepcopy(self.placement)
        for p in proposals:
            if p['speedup'] < cur_p['speedup']:
                continue
            if p['waste'] > self._max_waste:
                continue
            if sum(list(p['placement'].values())) > 2 * sum(list(cur_p['placement'].values())):
                continue
            #if sum(list(p['placement'].values())) <= sum(list(cur_p['placement'].values())):
            #    continue
            for gpu in self.application.gpu_types:
                #gpu_types = list(copy.deepcopy(self.application.gpu_types))
                #gpu_types.pop(gpu)
                #other_gpus = gpu_types

                #if (p['placement'][gpu] > cur_p['placement'][gpu]) :
                if judge(gpu, p, cur_p):
                    tmp = copy.deepcopy(p)
                    tmp_placement = copy.deepcopy(cur_p['placement'])
                    tmp_placement.update({gpu: p['placement'][gpu]})
                    tmp.update({'placement': tmp_placement})

                    inc_gpu = gpu
                    inc_num = p['placement'][gpu] - cur_p['placement'][gpu]
                    inc_perf = ((p['speedup'] / cur_p['speedup']) - 1) / inc_num + 1

                    tmp.update({'inc_gpu': inc_gpu})
                    tmp.update({'inc_num': inc_num})
                    tmp.update({'inc_perf': inc_perf})

                    if sum(list(tmp_placement.values())) <= 2 * sum(list(cur_p['placement'].values())):
                        available_proposals[gpu].append(tmp)

        submitted = []
                
        for gpu, increment_proposals in available_proposals.items():
            if len(increment_proposals) > 0:
                increment_proposals = sorted(increment_proposals, key=lambda x:x['inc_perf'], reverse=True)

                #logger.critical("### Submit proposals {}--{}".format(gpu, increment_proposals[0]))
                submitted.append(increment_proposals[0])
        
        #logger.critical(submitted)
        return submitted, cur_p
    





    def step(self, seconds):
        if not self.placement:
            # No resources are allocated to this job.
            self.current_time += seconds
            return
        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        #self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        seconds -= delay
        if seconds > 0 and self.completion_time is None:
            assert self.progress < self.total_progress
            # Calculate true (simulated) time.
            cur_throughput = self.pred_throughput(self.placement)
            step_time = self.total_batch_size / cur_throughput
            baseline = list(self.get_gpu_time().values())[-1]
            comp_time = baseline / self.placement['speedup']
            # Update the estimated throughput parameters.
            self.update_params(self.placement['placement'], self.placement['threads'], comp_time, step_time)

            remaining_time = float(self.total_progress-self.progress)/cur_throughput
            if remaining_time <= seconds:
                self.progress += cur_throughput * remaining_time
                #self.attained_service += remaining_time * sum(self.placement)
                self.completion_time = self.current_time + remaining_time
                self.current_time += seconds
                self.placement = dict()
            else:
                self.progress += cur_throughput * seconds
                #self.attained_service += seconds * sum(self.placement)
                self.current_time += seconds

    def reallocate(self, placement):
        if placement:
            #self.placement = tuple(placement)
            self.placement = placement
            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1
        else:  # De-allocate all resources.
            self.placement = ()


class Cluster(object):
    def __init__(self, workload, policy):
        self.workload = workload
        self.policy = policy
        self.current_time = 0

        self.gpu_types = ['v100', 'p100', 't4']
        self.gpu_nums = {'v100':32, 'p100':16, 't4':16}

        #if isinstance(policy, EasyPolicy) or isinstance(policy, FifoPolicy):
        self.jobs = OrderedDict((row.name, Job(row.name, APPLICATIONS[row.application], row.time,
                                                max_parallelism=row.max_parallelism,
                                                batch_size=row.batch_size,
                                                heterogeneous_deterministic=bool(row.enable_heterogeneous_deterministic),
                                                #min_parallelism=row.min_parallelism,
                                                epochs=row.epochs,
                                                preferred_gpu=row.preferred_gpu,
                                                gpu_nums=self.gpu_nums))
                                            for row in workload.itertuples() )
        # allocations save the accepted proposals of the running jobs
        self.allocations = OrderedDict()
        self.logs = []
        self.utility = []
        self.used_gpus = []
    
    def get_used_gpu_nums(self, allocations=None):
        if not allocations:
            allocations = self.allocations
        used = (0 for _ in self.gpu_types)
        for name, proposal in allocations.items():
            if proposal:
                placement = list(proposal['placement'].values())
                used = [a+b for a,b in zip(used, placement)]
        #remaining = [a-b for a,b in zip(self.gpu_nums.values(), used)]
        return dict(zip(self.gpu_types, used))

    def step(self, seconds=60):
        for job in self.jobs.values():
            job.step(seconds)
            #print("#### job name {}, cur_time {}, progress {}/{}".format(job.name, job.current_time, job.progress, job.total_progress))
        self.current_time += seconds

        if self.jobs: # self.job is not None, so there are remaining jobs
            # Optimize allocations.
            active_jobs = self.get_active_jobs()
            allocations = self.policy.optimize(active_jobs, self.gpu_nums)
            used_gpus = self.get_used_gpu_nums(allocations)
            for job in active_jobs.values():
                if (allocations.get(job.name) != self.allocations.get(job.name)):
                    proposal = allocations.get(job.name, dict())
                    job.reallocate(proposal)
                else:
                    assert "allocations error in simulator step"
                    #allocations.update({job.name: job.placement})
            self.allocations = allocations
            self.used_gpus = used_gpus

        if self.jobs:
            active_jobs = self.get_active_jobs()
            assert all(job.current_time == self.current_time for job in active_jobs.values())
        
        self.logs.append({
            "timestamp": self.current_time,
            "gpu_nums": self.used_gpus,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "total_progress": job.total_progress,
                    "num_restarts": job.num_restarts,
                    "placement": str(job.placement),
                    "batch_size": job.batch_size,
                    "submission_time": job.submission_time,
                    "completion_time": job.completion_time,
                    # "grad_params": job.grad_params,
                }
                for job in self.jobs.values() if job.submission_time <= self.current_time
            ],
        })

    def get_active_jobs(self):
        active = OrderedDict()
        for name, job in self.jobs.items():
            if job.submission_time <= self.current_time and job.completion_time is None:
                active[name] = job
        return active

    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs.values())

    def output_logs(self, path):
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")

    def get_jcts(self):
        return {
            val["name"]: val["completion_time"] - val["submission_time"]
            for val in self.logs[-1]["submitted_jobs"]
            if val["completion_time"] is not None
        }

def simulate(args):
    workload = pandas.read_csv(args.workload)
    if args.policy == "easyscale":
        policy = EasyPolicy()
    if args.policy == "easyscale_homo":
        policy = EasyPolicy(force_homo=True)
    if args.policy == "fifo":
        policy = FifoPolicy()
    if args.policy == "srtf":
        policy = SrtfPolicy()
    if args.policy == "tiresias":
        policy = TiresiasPolicy()
    simulator = Cluster(workload, policy)
    while not simulator.all_complete():
        simulator.step(args.interval)
        print("---------------- SIMULATOR TIME: {} ----------------"
              .format(simulator.current_time))
        print("Active jobs:")
        for val in simulator.logs[-1]["submitted_jobs"]:
            if val["submission_time"] <= simulator.current_time and val["completion_time"] is None:
                print("    {}:\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[placement {}]".format(
                      val["name"], val["epoch"], val["num_restarts"], val["batch_size"], val["placement"]))
        print("GPU utilization: {}/{}".format(simulator.used_gpus, simulator.gpu_nums))
        print("Completed jobs:")
        jct_dict = simulator.get_jcts()
        print(list(jct_dict.keys()))
        print("Average JCT:", sum(jct_dict.values()) / len(jct_dict) if jct_dict else 0)
        print("\n")
    if args.output:
        simulator.output_logs(args.output)
    return simulator.logs, simulator.get_jcts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument("--policy", type=str, default="easyscale",
                        choices=["fifo", "srtf", "tiresias", "optimus", "pollux", "easyscale", "easyscale_homo"])
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--num-nodes", type=int, default=8,
                        help="number of nodes in cluster")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="number of GPUs per node")
    parser.add_argument("--low-util", type=float,
                        help="low utility threshold")
    parser.add_argument("--high-util", type=float,
                        help="high utility threshold")
    parser.add_argument("--output", type=str,
                        help="path to output logs")
    args = parser.parse_args()
    if os.path.isdir(args.workload):
        assert args.output is not None and os.path.isdir(args.output)
        args_list = []
        for workload in glob.glob(args.workload + "/*.csv"):
            name = os.path.basename(workload)[:-4]
            args_list.append(copy.deepcopy(args))
            args_list[-1].workload = workload
            args_list[-1].output = args.output + "/" + name + ".log"
        with multiprocessing.Pool(processes=8) as pool:
            ret_list = pool.map(simulate, args_list)
        summary = {"jcts": {}, "avgs": {}}
        for args_item, (_, jct_dict) in zip(args_list, ret_list):
            name = os.path.basename(args_item.workload)[:-4]
            summary["jcts"][name] = jct_dict
            summary["avgs"][name] = sum(jct_dict.values()) / len(jct_dict)
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])
        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
    else:
        simulator_logs, jcts = simulate(args)

        with open("./summary_{}.json".format(args.policy), "w") as f:
            json.dump(simulator_logs, f, indent=4)
