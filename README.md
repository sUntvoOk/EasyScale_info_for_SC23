# Artifact for EasyScale (SC23)

## Environments

We have built a small cloud cluster with 4 servers, including 1 pure CPU server (as the master node), 1 server equipped with 8 V100 GPUs, 1 server equipped with 8 P100 GPUs, and 1 server equipped with 4 T4 GPUs.

**We have already prepared the environment on the cluster. Therefore, the artifact evaluator(s) do not need do that again.** The following steps is only provided to show our preparing procedure. 

All the following steps are conducted on a master node. 

* Configure the SSH login without password.
* Mount the NAS storage. The NAS is shared by all servers in the cluster, and stores all artifact codes.
* Install docker and nvidia-docker on the GPU servers.
* Build the docker container named as *easyscale*.

## Reproducing Experiement Results

### Accuracy-consistency (Figure 9)

We demonstrate the accuray-consistency of EasyScale here. 
The baseline is PyTorch DDP (under two configurations: {homo} and {heter}) with 4 V100 GPUs, and each DDP worker trains 300 mini-batches.
We run EasyScale (under four configurations: {D0}, {D1}, {D0+D2}, {D1+D2}) in three difference stages, specifically:
- stage 0: 4 V100 (each V100 holds 1 EST(s))
- stage 1: 2 V100 (each V100 holds 2 EST(s))
- stage 2: 1 V100 and 2 P100 (each v100 holds 2 EST(s), each P100 holds 1 EST(s))

In each stage, each EST trains 100 mini-batches, and totally 100\*3=300 mini-batches in all stages.
We compare the loss values of the last DDP worker and the last EST in each mini-batch. 

* Train with EasyScale-D0, EasyScale-D1, EasyScale-D0+D2, EasyScale-D1+D2, DDP-homo, and DDP-heter, and collect the logs:
```
# On the master node
cd easyscale-artifact/accuracy-consistency
python3 run.py
```

* Plot the figure:
```
# On the master node
cd plot
python3 plot.py
```

* Please see  `resnet50_homo.png`, `shufflenetv2_homo.png` (Figure 9(a)) and `resnet50_hete.png`, `shufflenetv2_hete.png` (Figure 9(b))

* The loss curves of EasyScale-D1 should be exactly the same as the of DDP-homo in stage 0 and stage 1.
The loss curves of EasyScale-D1+D2 should be exactly the same as the of DDP-heter in all stages (i.e., 0, 1, 2).
These results indicate the accuracy-consistency of EasyScale.

### Efficient GPU resource sharing (Figure 10)

This experiment is used to demonstrate advantage of EasyScale on GPU memory usage and training throughput by comparing with worker packing (implemented in PyTorch DDP).

* SSH login into a V100 server, and then switch to the easyscale container:
```
ssh v0
docker exec -ti easyscale bash
```

* Run EasyScale and worker packing on ResNet50 and ShuffleNetv2, and the corresponding output logs are generated in individual folders:
```
cd /workspace/efficient-gpu-sharing
bash run_run.sh
```

* Plot the figure:
```
cd /workspace/efficient-gpu-sharing/plot
# Process all the logs and generate the final memory/throughput numbers in CSV
bash process_log.sh
# Plot
python3 plot.py
```

* Please see `resnet50.png` (Figure 10(a)) and `shufflenetv2.png` (Figure 10(b))

* For the result description, please refer to Section 5.1.2


### Context switching overhead of EST (Figure 11) & Gradient copy/sync overhead of EST (Figure 13)

This experiment is used to study the overhead of context switching and gradient copy/sync of EST in EasyScale. 

* SSH login into a V100 server, and then switch to the easyscale container:
```
ssh v0
docker exec -ti easyscale bash
```

* Run EasyScale and DDP on all workloads, and the corresponding output logs are generated in individual folders:
```
cd /workspace/EST-overhead
bash run_run.sh
```

* Plot the figure:
```
cd /workspace/EST-overhead/plot
# Process all the logs and generate the final execution time numbers in CSV
bash process_log.sh
# Plot
python3 plot.py
```

* Please see `context_switch.png` (Figure 11) and `gradient_worker.png` (Figure 13)

* For the result description, please refer to Section 5.1.2 and Section 5.1.3


### Overall overhead of EasyScale (Figure 12)

This experiemnt is complicated and needs to compile many versions of our modified PyTorch. Therefore, we just provide our source data and plotting scripts here.

```
# On the master node
cd easyscale-artifact/overall-overhead
python3 plot.py
```

* Please see `overhead.png` (Figure 12)

* For the result description, please refer to Section 5.1.2


### Trace experiment (Figure 14 and 15)

The `simulator` folder contains the code used for the simulater-based trace experiment in Section 5.2. In particular, you can use the codes to run the simulator on the 8 workloads (in Table 1), to reproduce the experiments described by Figure 14 and Figure 15.

<!--
The contents are summarized as follows:

- **easy_policy.py** contains the implementation of EasyScale's intra-job policy and inter-job policy.
- **TODO**
-->

* Collect running logs of the workloads with all valid resource configurations (which is generated by the intra-job scheduler), so as to replaying them for the simulated jobs:

**Note that this step takes a relatively long time, as each workdload can be executed for many times under different resource configurations. So we provide our logs in *log* folder.**

<!--
```
# On the master node
cd easyscale-artifact/scripts
# Collect log
python3 step1_run_proposals.py
# Process the logs and transform them into JSON files
python3 step2_parse_result.py
```
-->

* Run the simulation about YARN-CS, EasyScale-homo, and EasyScale-heter policies. 
Specifically, the scheduling interval can be set using the `--interval` flag (in seconds, we use the default value of 60). You can increase the number of interval for faster simulation at the expense of decreased simulating quality.

```
# On the master node
cd /mnt/cloud/ADAE/easyscale-artifact/simulator
# Run simulator
bash run.sh
```

* Plot the figure:
```
cd /mnt/cloud/ADAE/easyscale-artifact/simulator/plot
# Process the simulation logs and transform them into CSV file:
python process_log.py
# Plot Figure 14
gnuplot jct_makespan.plt
# Plot Figure 15
gnuplot trace_alloc.plt
```

* Please see `jct_makespan.png` (Figure 14) and `trace_alloc.png` (Figure 15)

* In `jct_makespan.png`, EasyScale has lower makespan and JCT than YARN-CS, indicating EasyScale can improve the aggregated job throughput. 
In `trace_alloc.png`, the GPU resource allocation curve of EasyScale_{hete} is higher than EasyScale_{homo}, indicating EasyScale can better utilize the heterogeneous GPU resources.

<!--
* End-to-end script, all the above steps can be exected by the below single script:
```
bash reproduce_simulator_figure.sh
```
-->
