import collections
from collections import OrderedDict
import functools
from functools import reduce
import glob
import math
import copy
import random
import os, sys
import pandas
import numpy as np

from logger import my_logger
logger = my_logger


class FifoPolicy(object):
    def __init__(self):
        pass

    # 优先调度提交时间最早的job
    def optimize(self, jobs, gpus):
        # The jobs here are active jobs
        allocations = OrderedDict()
        logger.debug('## policy optimize input : gpus {}'.format(gpus))
        remaining_gpus = list(gpus.values())
        for job in jobs.values():
            logger.critical("## job {}, placement {}".format(job.name, job.placement))
            #allocations[job.name] = job.get_placement_proposals()[0]
            proposals = job.get_placement_proposals(force_non_elastic=True)
            # 一共只有一个proposal
            assert len(proposals) == 1
            #proposals = sorted(proposals, key=lambda x:x['speedup'], reverse=True)

            allocations[job.name] = proposals[0]
            break

        return allocations
            

            #for p in proposals:
            #    remaining_tmp = [a-b for a,b in zip(remaining_gpus, list(p['placement'].values()))]
            #    if all(r >= 0 for r in remaining_tmp):        
            #        allocations[job.name] = p
            #        remaining_gpus = remaining_tmp
            #        
            #        return allocations

        #return allocations


class SrtfPolicy(object):
    def __init__(self):
        pass

    def optimize(self, jobs, gpus):
        # The jobs here are active jobs
        allocations = OrderedDict()
        logger.debug('## policy optimize input : gpus {}'.format(gpus))
        remaining_gpus = list(gpus.values())

        all_proposals = []
        for job in jobs.values():
            logger.debug("## job {}, placement {}".format(job.name, job.placement))
            #allocations[job.name] = job.get_placement_proposals()[0]
            proposals = job.get_placement_proposals(force_non_elastic=True)
            # 一共只有一个proposal
            assert len(proposals) == 1

            p = copy.deepcopy(proposals[0])
            rt = (job.total_progress - job.progress) / job.pred_throughput(p)
            p.update({'job_name': job.name})
            p.update({'remaining_time': rt})        

            all_proposals.append(p)
        
        all_proposals = sorted(all_proposals, key=lambda x:x['remaining_time'], reverse=False)

        for p in all_proposals:
            remaining_tmp = [a-b for a,b in zip(remaining_gpus, list(p['placement'].values()))]
            if all(r >= 0 for r in remaining_tmp):        
                allocations[p['job_name']] = p
                remaining_gpus = remaining_tmp

        for alloc in allocations.values():
            try:
                alloc.pop('job_name')
                alloc.pop('remaining_time')
            except Exception:
                print(Exception)

        return allocations


## Modified from pollux artifact
class TiresiasPolicy(object):
    def __init__(self, time_fn):
        self._time_fn = time_fn
        self._queue_threshold = 3600 * 16
        self._solve_starvation = 0
        self._queue_0 = []
        self._queue_1 = []
        self._status = {}
        self._last_check_time = collections.Counter()
        self._total_executed_time = collections.Counter()
        self._executed_time = collections.Counter()
        self._last_pending_time = collections.Counter()
        self._pending_time = collections.Counter()

    def optimize(self, jobs, nodes, prev_allocations, node_template):
        event_time = int(self._time_fn())
        # Remove completed jobs.
        self._queue_0 = [key for key in self._queue_0 if key in jobs]
        self._queue_1 = [key for key in self._queue_1 if key in jobs]
        self._status = {key: val for key, val in self._status.items() if key in jobs}
        allocations = {key: val for key, val in prev_allocations.items() if key in jobs}
        # Add new jobs to pending.
        for key, job in jobs.items():
            if key not in self._status:
                self._status[key] = 'PENDING'
                self._queue_0.append(key)
        # Update queues.
        for key, job in jobs.items():
            assert self._status[key] in ('RUNNING', 'PENDING')
            if self._status[key] == 'RUNNING':  # Job is running.
                tmp = int(event_time - self._last_check_time[key]) 
                self._total_executed_time[key] = int(self._total_executed_time[key] + tmp)
                self._executed_time[key] = int(self._executed_time[key] + tmp) # decide job priority queue
                self._last_check_time[key] = event_time
                # check demotion
                gputime = int(self._executed_time[key] * job.max_replicas)
                if key in self._queue_0 and gputime >= self._queue_threshold:
                    self._queue_0.pop(self._queue_0.index(key))
                    self._queue_1.append(key)
                    print("job {} demote to Q1".format(key))
            elif self._status[key] == 'PENDING':
                tmp = int(event_time - self._last_check_time[key]) 
                self._last_check_time[key] = event_time
                self._pending_time[key] = int(self._pending_time[key] + tmp) #this is the total pending_time
                if self._executed_time[key] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    self._last_pending_time[key] = int(self._last_pending_time[key] + tmp) #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned 
                if self._solve_starvation > 0 and key not in self._queue_0 and \
                        self._total_executed_time[key] > 0 and self._executed_time[key] > 0:
                    if self._last_pending_time[key] >= int(self._executed_time[key] * self._solve_starvation):
                        self._executed_time[key] = 0
                        self._last_pending_time[key] = 0
                        self._queue_0.append(key)
                        self._queue_1.pop(self._queue_1.index(key))
        # Update statuses.
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        num_gpus = sum(total_gpus.values())
        for queue in (self._queue_0, self._queue_1):
            for idx in queue:
                if jobs[idx].max_replicas <= num_gpus:
                    self._status[idx] = 'RUNNING'
                    num_gpus -= jobs[idx].max_replicas
                else:
                    self._status[idx] = 'PENDING'
                    allocations.pop(idx, None)
        # Update allocations.
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        for queue in (self._queue_0, self._queue_1):
            for idx in queue:
                if self._status[idx] == 'RUNNING' and not allocations.get(idx):
                    # Allocate resources.
                    allocations[idx] = []
                    while len(allocations[idx]) < jobs[idx].max_replicas:
                        node_idx, count = free_gpus.most_common(1)[0]
                        num = min(count, jobs[idx].max_replicas - len(allocations[idx]))
                        allocations[idx].extend([node_idx] * num)
                        free_gpus[node_idx] -= num
        # Objective values, allocations, active nodes.
        return allocations, len(nodes)


## Modified from pollux artifact
class OptimusPolicy(object):
    def __init__(self):
        pass

    def optimize(self, jobs, nodes, prev_allocations, node_template):
        allocations = {k: v for k, v in prev_allocations.items() if k in jobs}
        for job in jobs.values():
            completion_epoch = job.application.get_completion_epoch(
                    job.target_batch_size)
            if completion_epoch <= job.epoch:
                job.remaining = 1
            else:
                job.remaining = (job.application.get_iteration(job.target_batch_size, completion_epoch) -
                                 job.application.get_iteration(job.target_batch_size, job.epoch))
        min_replicas = {}
        for key, job in jobs.items():
            min_replicas[key] = 1  # math.ceil(job.target_batch_size / job.application.max_local_bsz)
        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        num_replicas = {}
        gain = {}
        for key, job in sorted(jobs.items(), key=lambda item: min_replicas[item[0]]):
            if min_replicas[key] > num_gpus:
                num_replicas[key] = 0
                gain[key] = 0
                continue
            num_replicas[key] = min_replicas[key]
            num_gpus -= min_replicas[key]
            if num_replicas[key] + 1 > job.max_replicas or num_gpus < 1:
                gain[key] = 0
            else:
                gain[key] = (self.predict_step_time(job, num_replicas[key]) -
                             self.predict_step_time(job, num_replicas[key] + 1)) * job.remaining
        # Add resources in order of maximum marginal gain.
        while num_gpus > 0 and max(gain.values()) > 0:
            key = max(gain, key=lambda k: gain[k])
            job = jobs[key]
            num_replicas[key] += 1
            if num_replicas[key] + 1 > job.max_replicas or num_gpus < 1:
                gain[key] = 0
            else:
                gain[key] = (self.predict_step_time(job, num_replicas[key]) -
                             self.predict_step_time(job, num_replicas[key] + 1)) * job.remaining
            num_gpus -= 1
        # Placements.
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}
        job_keys = sorted(jobs, key=lambda k: num_replicas[k])
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        for key in job_keys:
            if num_replicas[key] > 0 and not allocations.get(key):
                # Allocate resources.
                allocations[key] = []
                while len(allocations[key]) < num_replicas[key]:
                    node_idx, count = free_gpus.most_common(1)[0]
                    num = min(count, num_replicas[key] - len(allocations[key]))
                    allocations[key].extend([node_idx] * num)
                    free_gpus[node_idx] -= num
        return allocations, len(nodes)

    def predict_step_time(self, job, num_replicas):
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        local_bsz = math.ceil(job.target_batch_size / num_replicas - 1e-8)
        accum_steps = math.ceil(local_bsz / job.application.max_local_bsz - 1e-8) - 1
        if num_replicas == 1:
            accum_steps = max(1, accum_steps)
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = num_replicas * (accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(job.application.max_batch_size / count))
        #throughput = job.speedup_fn._goodput_fn.throughput(len(placement), num_replicas, atomic_bsz, accum_steps)
        #return atomic_bsz * count / throughput
        step_time, sync_time = job.application.get_throughput(placement, atomic_bsz)
        return step_time + (step_time - sync_time) * accum_steps
