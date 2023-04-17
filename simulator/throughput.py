import autograd
import numpy as np
import collections
import scipy.optimize
import scipy.stats

PerfParams = collections.namedtuple("PerfParams", [
    "c",
    "z",
    "per_bw", 
    "gamma",
])


# 0: v100, 1: p100, 2: t4
## TODO: 如果只有一张卡，那么是不需要进行通信的。暂时没有考虑到这种情况。
def _predict_step_time(params, placements, max_threads, comp_times):
    params = PerfParams(*params)

    # overlapped percentage
    per = params.per_bw/max_threads
    # communication time
    comm_time = _predict_comm_time(params, placements)
    # un-overlapped computation time
    t_un = (1.0 - per) * comp_times
    # total time of overlapped allreduce communication and backward computation
    #t_ol_log = np.log((per * comp_times) ** params.gamma + comm_time ** params.gamma) / params.gamma
    #t_ol = np.exp(t_ol_log)
    t_ol = ((per * comp_times) ** params.gamma + comm_time ** params.gamma) ** (1/params.gamma)
    #print("t_un is ", t_un)
    #print("t_ol is ", t_ol)
    # total time
    t = t_un + t_ol
    
    return t

def _predict_comm_time(params, placements):
    z = PerfParams(*params).z
    c = PerfParams(*params).c
    comm_time = z + c * np.maximum(placements, 1e-8)
    #comm_time = z + c * np.maximum(np.log2(placements), 1e-8)
    #comm_time = z + c * np.maximum(np.ceil(placements/4), 1e-8)
    return comm_time


def _rmse(pred, true):
    return np.sqrt((((pred - true)) ** 2).mean())

def _obj_fn(params, placements, max_threads, comp_times, total_times):
    params = PerfParams(*params)
    pred_t = _predict_step_time(params, placements, max_threads, comp_times)

    # RMSLError of step time predictions.
    err = _rmse(pred_t, total_times)
    #print("## err={}".format(err))
    # L2 regularization towards a smaller gamma, because it's easier to
    # optimize the alpha and beta parameters when gamma is smaller.
    #reg1 = 1e-3 * (params.gamma - 1) ** 2
    # Penalize retrogression terms to prefer a more optimistic model.
    #reg2 = 1e-2 * ((params.c / params.z) ** 2 )
    return err #+ reg1 + reg2


def fit_perf_params(placements, threads, comp_times, total_times):   
    # Fit the performance model.

    # HACK: We want to use the original numpy module for calls from the
    # SpeedupFunction for performance reasons, but also need those functions to
    # use autograd.numpy when we want to differentiate them. We patch the
    # global np reference only for the code invoked rom this function.
    global np  # Replace numpy from autograd.
    orig_np = np
    np = autograd.numpy

    #placements = np.array(tuple(placements))
    placement_0s = np.array([a[0] for a in placements])
    placement_1s = np.array([a[1] for a in placements])
    placement_2s = np.array([a[2] for a in placements])
    #threads = np.array(tuple(threads))
    max_threads = np.array([max(t) for t in threads])
    comp_times = np.array(comp_times)
    total_times = np.array(total_times)

    # Set initial params to reasonable values.
    params = [1e-2]  + [1e-1]   + [0.5] + [1 + 1e-3]
    # Set lower/upper bounds for each parameter. Add a small slack to lower
    # bounds to avoid numerical instability issues.
    lower = [1e-8]   + [1e-8]   + [1e-5]  + [1.0]
    upper = [np.inf] + [np.inf] + [1.0]  + [10.0]
    bounds = scipy.optimize.Bounds(lower, upper, keep_feasible=True)
    #args = (placements, threads, comp_times, total_times)
    args = (placement_0s + placement_1s + placement_2s, max_threads, comp_times, total_times)
    # FIXME: need to handle optimization failures and propagate to the Trainer.
    grad_fn = autograd.grad(_obj_fn)
    result = scipy.optimize.minimize(_obj_fn, params, args=args,
                                     jac=grad_fn, bounds=bounds)
    params = result.x
    
    np = orig_np  # Restore original numpy.

    print(result.items())
    return PerfParams(*params)


def predict_step_time(params, placement, thread, comp_time):   
    params = PerfParams(*params)

    placement_0 = np.array([ list(placement.values())[0] ])
    placement_1 = np.array([ list(placement.values())[1] ])
    placement_2 = np.array([ list(placement.values())[2] ])
    max_thread  = np.array([ max(list(thread.values())) ])
    comp_time   = np.array([ comp_time ])

    placement = placement_0 + placement_1 + placement_2

    step_times = _predict_step_time(params, placement, max_thread, comp_time)

    return float(step_times[0])


class ThroughputFunction(object):

    def __init__(self, perf_params):
        self._perf_params = PerfParams(*perf_params)

    def __call__(self, placement, thread, comp_time, total_batch_size):
        return self.get_throughput(placement, thread, comp_time, total_batch_size)
    
    #def __call__(self, placement, thread, comp_time):
    #    return self.get_time(placement, thread, comp_time)

    def get_time(self, placement, thread, comp_time):
        #return _predict_step_time(self._perf_params, placement, thread, comp_time)
        return predict_step_time(self._perf_params, placement, thread, comp_time)

    def get_throughput(self, placement, thread, comp_time, total_batch_size):
        t = self.get_time(placement, thread, comp_time)
        return total_batch_size / t
