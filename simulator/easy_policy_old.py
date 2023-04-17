import collections
from collections import OrderedDict
import functools
from functools import reduce
import itertools
import glob
import math
import copy
import random
import os, sys
import pandas
import numpy as np

#import pdb

from logger import my_logger
logger = my_logger


class Intrajob(object):
    def __init__(self, max_parallelism, gpu_types, gpu_flops, gpu_perf, gpu_mems, gpu_nums, interference=None):
        assert len(gpu_types) == len(gpu_flops) == len(gpu_perf) == len(gpu_mems) == len(gpu_nums)
        
        self.mp = max_parallelism   # e.g., 16
        self.types = tuple(gpu_types)
        self.flops = tuple(gpu_flops)
        self.perf = tuple(gpu_perf) # e.g., [3,2,1]
        self.mems = tuple(gpu_mems)
        self.nums = tuple(gpu_nums)
        self.processes = tuple([1 for i in range(len(self.types))])
        
        self.enable_heterogenous_mem = False
        if interference:
            self.enable_heterogenous_mem = True
            self.interference = tuple(interference)
            # determine processes according to mems
            min_mem = min(self.mems)
            self.processes = tuple([m//min_mem for m in self.mems])
            # determine gpu_perf under interference
            self.perf = tuple([a*b for a,b in zip(self.perf, self.interference)])

        self.proposals = dict()

    def get_perf_candidates(self):
        logger.debug('## INPUT: mp = {}'.format(self.mp))
        logger.debug('## INPUT: GPU perf is {}'.format(self.perf))

        perf_combinations = []
        for m in range(1, self.mp+1):
            logger.debug('## m is {}'.format(m))
            perf_scaled = tuple([p*m for p in self.perf])
            
            perf_varients = [] 
            for ps in perf_scaled:
                v = []
                for i in [math.floor(ps), math.ceil(ps)]:
                    if i not in v:
                        v.append(int(i))
                perf_varients.append(v)
            logger.debug('## all perf varients are {}'.format(perf_varients))

            temp = list(itertools.product(*perf_varients))
            for p in temp:
                perf_combinations.append( (p, m) )
        logger.info('## all perf combinations are {}'.format(perf_combinations))

        perf_candidates = []
        for p, m in perf_combinations:
            perf_dict = {a:b for a,b in zip(self.types, p)}
            for i in range(2, len(self.types)+1):
                comb = itertools.combinations(perf_dict.items(), i)
                for j in comb:
                    # prune
                    if all(v > self.mp for k,v in j): continue
                    tmp = {a:0 for a in self.types}
                    for k,v in j: tmp[k] = v
                    perf_candidates.append( (tuple(tmp.values()), m) )

        # add situation: only use one gpu type
        for i,_ in enumerate(self.types):
            for m in range(1, self.mp+1):
                tmp = [0 for i in self.types]
                tmp[i] = m
                perf_candidates.append( (tuple(tmp), m) )

        perf_candidates = list(set(perf_candidates))
        self.perf_candidates = tuple(perf_candidates)
        logger.info('## all perf combinations (final) are {}'.format(perf_candidates))

    def get_proposals(self):
        proposals = []
        for perf, m in self.perf_candidates:
            logger.info('## current perf candidate {}, nums {}'.format(perf, self.nums))

            combinations = self.combinationSum(perf = perf)
            logger.info('## combinations {}'.format(combinations))
            perf_scaled = tuple([p*m for p in self.perf])
            for comb in combinations:
                logger.info("## perf_scaled {}, perf {}, comb {}".format(perf_scaled, perf, comb))
                w, threads = self.waste(theory=perf_scaled, actual=perf, comb=comb, mp=self.mp, processes=self.processes)
                logger.info("## w {}, threads {}".format(w, threads))
                proposals.append([comb, w, threads, self.mp])
            logger.info("## len(proposals) is {}".format(len(proposals)))
        proposals = sorted(proposals, key=lambda x:x[1])
        #proposals = sorted(proposals, key=lambda x:x[0][0], reverse=False)

        return self._convert_to_final_proposals(proposals)

    #def combinationSum(candidates: List, target: int) -> List[List]:
    def combinationSum(self, perf):
        nums = self.nums
        target = self.mp
        target = int(math.ceil(target))
        logger.debug("## combinationSum target {}".format(target))
        
        def expand(perf, nums):
            assert len(perf) == len(nums)
            val = []
            idx = []
            for i in range(len(perf)):
                if perf[i] > 0:
                    val += [perf[i]] * nums[i]
                    idx += [i] * nums[i]
            return list(zip(val, idx))
        
        candidates = expand(perf, nums)
        candidates = sorted(candidates, key=lambda x:x[0], reverse=True) 
        # sort numbers to break the search while target < all the values
        logger.info("## candidates are {}".format(candidates))
        processes = self.processes
        
        res = []
        temp = []
    
        def dfs(candidates, target, processes):
            n = len(candidates)
            for i in range(n):
                gpu_idx = candidates[i][1]
                if target > 0 and target <= candidates[i][0]*processes[gpu_idx]:
                    res.append(temp.copy()+[candidates[i]])
                    break
                if i>0 and candidates[i]==candidates[i-1]:
                    continue 
                temp.append(candidates[i])
                dfs(candidates[i+1:], target-candidates[i][0]*processes[gpu_idx], processes) 
                # use candidates[i+1:] to avoid reuse of the same element in candidates
                temp.pop()
        dfs(candidates, target, self.processes)

        #def dfs(begin, target):
        #    if target <= 0:
        #        res.append(temp[:])
        #    for i in range(begin, len(candidates)):
        #        gpu_idx = candidates[i][1]
        #        if candidates[i][0]*processes[gpu_idx] > target:
        #            break
        #        if i > begin and candidates[i] == candidates[i-1]:
        #            continue
        #        temp.append(candidates[i])
        #        dfs(begin+1, target-candidates[i][0]*processes[gpu_idx])
        #        temp.pop()
        #dfs(0, target)

    
        res = sorted(res, key=lambda x:len(x))
    
        ## TODO:: pruning
        #toRemove = []
        #for i in range(len(res)):
        #    for j in range(len(res)):
        #        if j > i and set(res[j]).issuperset(set(res[i])):
        #            toRemove.append(res[j])
        #assert len(toRemove) >= 0
    
        #for r in toRemove:
        #    if r in res:
        #        res.remove(r)
        return res

    def _convert_to_final_proposals(self, proposals):
        merged = []
        for p in proposals:
            # p[0]: comb; p[1]: waste; p[2]: threads; p[3]: mp
            cnt = [0 for i in range(len(self.types))]
            for c in p[0]:
                idx = c[1]
                cnt[idx] += 1
            speedup = [a*b for a,b in zip(self.perf, cnt)]
            speedup = sum(speedup) * (1.0 - p[1])
            flops = [a*b for a,b in zip(self.flops, cnt)]
            flops = sum(flops) * (1.0 - p[1])
            merged.append([cnt, p[1], p[2], flops, speedup, p[3]])

        merged = sorted(merged, key=lambda x:x[0][0], reverse=True)
        merged = sorted(merged, key=lambda x:x[4], reverse=True)

        logger.info("## len(merged) {}".format(len(merged)))

        final = OrderedDict()
        for p in merged:
            key = tuple(p[0])
            #value = [p[1], p[2], p[3], p[4], p[5]]
            value = [vv for i,vv in enumerate(p) if i > 0]
            if final.__contains__(key):
                if p[1] < final[key][0] :
                    final[key] = value
            else:
                final[key] = value

        ret = []
        for k, v in final.items():
            placement = dict(zip(self.types, k))
            processes = [0 for i in self.types]
            for i in range(len(self.types)):
                if k[i] > 0:
                    processes[i] = self.processes[i]
            processes = dict(zip(self.types, processes))
            threads = dict(zip(self.types, v[1]))
            waste = float(v[0])
            flops = float(v[2])
            speedup = float(v[3])
            mp = int(v[4])

            proposal = {'placement':placement,
                        'processes':processes,
                        'threads':threads,
                        'waste':waste,
                        #'flops':flops,
                        'speedup':speedup,
                        'mp':mp }
            ret.append(proposal)

        return ret

    @staticmethod
    def waste(theory, actual, comb, mp, processes):
        logger.info('@@ waste input : theory {}, actual {}, comb {}, mp {}, processes {}'.format(theory, actual, comb, mp, processes))

        #threads = [0 for i in range(len(theory))]
        threads = list(actual[:])

        # count the number of each gpu type
        cnt = [0 for i in range(len(theory))]
        for c in comb:
            idx = c[1]
            cnt[idx] += 1

        scales_temp = [-float('inf') for i in range(len(theory))]
        for i in range(len(theory)):
            if cnt[i] == 0:
                scales_temp[i] = -float('inf')
            else:
                scales_temp[i] = actual[i] / theory[i]
        logger.info("## scales {}".format(scales_temp))
        
        # 如果只有1种GPU
        is_single_type = False
        cnt_gpu_nums = cnt.copy()
        logger.info("## cnt_gpu_nums {}".format(cnt_gpu_nums))
        if len(list(filter(lambda x: x>0, cnt_gpu_nums))) == 1:
            #print("\033[31;1mHello {}\033[0m".format(cnt_gpu_nums))
            is_single_type = True
            gpu_idx = 0
            for idx,_ in enumerate(cnt):
                if cnt[idx] > 0:
                    gpu_idx = idx
                    break
            logger.info("## gpu_idx {}".format(gpu_idx))

            # hack
            # correct the proposal when with one gpu type
            tmp_threads = cnt_gpu_nums[gpu_idx]*processes[gpu_idx]
            if mp % tmp_threads == 0: # one type and one gpu
                for idx,_ in enumerate(cnt):
                    if cnt[idx] > 0:
                        threads[idx] = mp // tmp_threads
                    else:
                        threads[idx] = 0
                return 0.0, threads
            else: # one type and multiple gpus
                w = 1.0 - mp / ( math.ceil(mp / tmp_threads) * tmp_threads )
                for idx,_ in enumerate(cnt):
                    if cnt[idx] > 0:
                        threads[idx] = int(math.ceil(mp / tmp_threads))
                    else:
                        threads[idx] = 0
                return w, threads

        std = max(scales_temp)
        ref = [float(a)/std for a in actual]

        logger.info("## ref is {}".format(ref))

        w = 0.0
        theory_allocated = 0.0
        actual_allocated = 0.0
        ref_allocated = 0.0
        for c in comb:
            idx = c[1]
            theory_allocated += theory[idx] * processes[idx]
            actual_allocated += actual[idx] * processes[idx]
            ref_allocated += ref[idx] * processes[idx]
            w += theory[idx] - ref[idx]

        logger.debug("## DEBUG: theory_allocated {}, actual_allocated {}, ref_allocated {}".format(theory_allocated, actual_allocated, ref_allocated))

        if actual_allocated >= mp:
            #w += (1.0 - mp/actual_allocated)*theory_allocated
            w += (actual_allocated - mp) / std
        else:
            logger.warning("## WARNING: actual_allocated must be larger than mp. Let waste=10000")
            w += 10000.0

        logger.debug("\t#### DEBUG: waste {:.2f}, allocated {:.2f}, {:.2f}%".format(w, theory_allocated, w/theory_allocated*100))

        return w/theory_allocated, threads


class EasyPolicy(object):
    def __init__(self, force_homo=False):
        # Waste thresholds for intra-job.
        self._max_waste = 0.30
        self.force_homo = force_homo

    def optimize(self, jobs, gpus):
        return self.optimize_v3(jobs, gpus)

    # 优先调度提交时间最早的job
    # 优先选择speedup最大的proposal
    def optimize_v0(self, jobs, gpus):
        # The jobs here are active jobs
        allocations = OrderedDict()
        logger.debug('## policy optimize input : gpus {}'.format(gpus))
        remaining_gpus = list(gpus.values())
        for job in jobs.values():
            logger.debug("## job {}, placement {}".format(job.name, job.placement))
            #allocations[job.name] = job.get_placement_proposals()[0]
            proposals = job.get_placement_proposals()
            #proposals = sorted(proposals, key=lambda x:x['flops'], reverse=False)
            proposals = sorted(proposals, key=lambda x:x['speedup'], reverse=True)
            proposals = sorted(proposals, key=lambda x:x['total_time'], reverse=False)


            for p in proposals:
                remaining_tmp = [a-b for a,b in zip(remaining_gpus, list(p['placement'].values()))]
                if all(r >= 0 for r in remaining_tmp):        
                    allocations[job.name] = p
                    remaining_gpus = remaining_tmp
                    break

        return allocations
    
    # 每次选择前k个job求局部最优，如果还有剩余GPU，那么继续选取k个job求局部最优
    def optimize_v1(self, jobs, gpus):
        # The jobs here are active jobs
        allocations = OrderedDict()
        logger.debug('## policy optimize input : gpus {}'.format(gpus))
        remaining_gpus = list(gpus.values())

        # Create the linear solver with the GLOP backend.
        #solver = pywraplp.Solver.CreateSolver('GLOP')
        
        window = 4

        # ss --> search space
        search_space = OrderedDict()
        for idx, job in enumerate(jobs.values()):
            if idx % window == 0:
                search_space = OrderedDict()
            proposals = job.get_placement_proposals()
            proposals = sorted(proposals, key=lambda x:x['flops'], reverse=False)
            proposals = sorted(proposals, key=lambda x:x['speedup'], reverse=True)
            
            sample_num = min(len(proposals), 10)
            search_space[job.name] = random.sample(proposals, sample_num)

            if idx % window == window - 1 or idx == len(jobs.values()) - 1:
                ss_len = (len(i) for i in search_space.values())
                ssk = list(search_space.keys())
                ssv = list(search_space.values())
                total_times = reduce(lambda x,y : x*y, ss_len)
                logger.debug("total_times {}".format(total_times))
                n = 0
                alloc = []
                while n < total_times:
                    tmp = n
                    buffer = []
                    indexs = []
                    for i in range(len(search_space)):
                        tmp, cur = divmod(tmp, len(ssv[i]))
                        buffer.append(ssv[i][cur])
                        indexs.append(cur)
                    speedup = reduce(lambda x,y : x+y, [p['speedup'] for p in buffer])
                    used_gpus = (0 for _ in remaining_gpus)
                    for p in buffer:
                        used_gpus = [a+b for a,b in zip(used_gpus, list(p['placement'].values()))]
                    indexs = tuple(indexs)
                    remaining_tmp = [a-b for a,b in zip(remaining_gpus, used_gpus)]
                    if all(r >= 0 for r in remaining_tmp):
                        alloc.append( {'proposal':indexs, 'speedup':speedup, 'used_gpus':used_gpus} )
                    n += 1 
                alloc = sorted(alloc, key=lambda x:x['speedup'], reverse=True)
                logger.debug("## alloc len {}".format(len(alloc)))
                assert len(alloc) > 0
                logger.critical(alloc[0]['proposal'])
                remaining_gpus = [a-b for a,b in zip(remaining_gpus, alloc[0]['used_gpus'])]
                for ijob, a in enumerate(alloc[0]['proposal']):
                    allocations[ssk[ijob]] = ssv[ijob][a]

                logger.debug("remaining gpus {}".format(remaining_gpus))
                if all(r == 0 for r in remaining_gpus):
                    break
                
        return allocations   


    def optimize_v3(self, jobs, gpus):
        # The jobs here are active jobs
        allocations = OrderedDict()
        logger.debug('## policy optimize input : gpus {}'.format(gpus))
        remaining_gpus = list(gpus.values())

        init_proposals = []
        all_proposals = []
        for job in jobs.values():
            if not job.placement:
                ### 只需要打开force_non_elastic 就可以关闭异构调度
                init_proposal = copy.deepcopy(job.get_placement_proposals(force_homo=self.force_homo)[-1])
                init_proposal.update({'job_name': job.name})
                init_proposals.append(init_proposal)
            else:
                proposals, cur = copy.deepcopy(job.submit_proposals())
                # 首先统计在不增量分配的情况下，剩余多少gpu
                remaining_tmp = [a-b for a,b in zip(remaining_gpus, list(cur['placement'].values()))]
                assert all(r >= 0 for r in remaining_tmp)
                remaining_gpus = remaining_tmp
                allocations[job.name] = cur 

                for p in proposals:
                    p.update({'job_name': job.name})
                    all_proposals.append(p)                

        all_proposals = sorted(all_proposals, key=lambda x:x['inc_perf'], reverse=True)

        #for p in all_proposals:
        #    logger.critical(p)
        
        # 然后给没有placement的任务分配资源
        processed_jobs = []
        for p in init_proposals:
            remaining_tmp = [a-b for a,b in zip(remaining_gpus, list(p['placement'].values()))]
            if all(r >= 0 for r in remaining_tmp) and p['job_name'] not in processed_jobs:
                allocations[p['job_name']] = p
                processed_jobs.append(p['job_name'])
                remaining_gpus = remaining_tmp

        for p in all_proposals:
            inc_placement = {k:0 for k in gpus}
            inc_placement.update({p['inc_gpu']: p['inc_num']})
            remaining_tmp = [a-b for a,b in zip(remaining_gpus, list(inc_placement.values()))]
            if all(r >= 0 for r in remaining_tmp) and p['job_name'] not in processed_jobs:
                allocations[p['job_name']] = p
                processed_jobs.append(p['job_name'])
                remaining_gpus = remaining_tmp
        
        #for job in jobs.values():
        #    if job.name not in processed_jobs:
        #        allocations[job.name] = cur_placement[job.name] 

        for alloc in allocations.values():
            try:
                alloc.pop('job_name')
                alloc.pop('inc_gpu')
                alloc.pop('inc_num')
                alloc.pop('inc_perf')
            except Exception:
                print(Exception)
                

        
        return allocations
            

if __name__ == '__main__':
    
    #intra = Intrajob(max_parallelism=16, gpu_types=['v100', 'p100', 't4'], gpu_flops=[15.7, 10.6, 8.1], gpu_perf=[3.1,2.4,1], gpu_nums=[0,0,16])
    #intra = Intrajob(max_parallelism=16, gpu_names=['v100', 'p100', 't4'], gpu_flops=[15.7, 10.6, 8.1], gpu_perf=[3.1,2.4,1], gpu_nums=[32,0,0])
    intra = Intrajob(max_parallelism=16, gpu_types=['v100', 'p100', 't4'], gpu_flops=[15.7, 10.6, 8.1], gpu_perf=[3.1,2.4,1], gpu_mems=[32,16,16], gpu_nums=[16,16,0], interference=[1,1,1])
    #intra = Intrajob(max_parallelism=16, gpu_names=['v100'], gpu_flops=[15.7], gpu_perf=[3.1], gpu_nums=[32])

    candidate = intra.get_perf_candidates()
    proposals = intra.get_proposals()

    for p in proposals:
        print(p)
    print(len(proposals))