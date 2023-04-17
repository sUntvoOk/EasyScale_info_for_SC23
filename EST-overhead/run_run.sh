#!/bin/bash

set -x
ulimit -c 0
export GPU_TYPE="V100"

export PYTORCH_GPU_VMEM=false
export EXPERIMENT_STEPS=100

# 最大bs：shufflenetv2: 256; resnet50: 128; vgg19: 64; ncf: 16384; yolov3: 8; bert: 16; electra: 16; swintransformer: 64

# !!!!!!!!!!!!!!!! 在log.sh的exp_dir中定义了log文件生成的目录，要挪位置记得改下

# mode=$1
# model=$2
# mp=$3
# pod=$4
# bs=$5

# 这样把运行不同的模型的命令拆开写可以很方便地选要跑哪些模型

# 目录名字参数：DDP_context_switch/EDDP_CPU_context_switch/EDDP_GPU_context_switch 
# 这个参数一个是根据前缀来选择跑ddp/eddp_cpu/eddp_gpu，另一个是可以按照需要取目录名，比如我想在不覆盖原来的实验的前提下跑多次实验对比结果，则可以改成DDP_context_switch1

export ARTIFACT_MODE="PROFILING"
export EXPERIMENT_STEPS=20
export PYTORCH_GPU_VMEM=false 

hd=0

#######################################################################################################################
### standard/baseline
## no heterogeneous 
## GEMM_ALGO not used actually

EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=4 ./test.sh  EDDP_CPU_standard vgg19 8 1 64
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=10 ./test.sh EDDP_CPU_standard yolov3 8 1 8
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_standard bert 8 1 16
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_standard swintransformer 8 1 64

EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=7 ./test.sh   EDDP_CPU_standard resnet50 8 1 128
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=7 ./test.sh   EDDP_CPU_standard shufflenetv2 8 1 256
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh  EDDP_CPU_standard ncf 8 1 16384
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh  EDDP_CPU_standard electra 8 1 16

HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=4 ./test.sh  DDP_standard vgg19 8 8 64
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=10 ./test.sh DDP_standard yolov3 8 8 8
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh DDP_standard bert 8 8 16
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh DDP_standard swintransformer 8 8 64

HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=7 ./test.sh   DDP_standard resnet50 8 8 128
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=7 ./test.sh   DDP_standard shufflenetv2 8 8 256
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh  DDP_standard ncf 8 8 16384
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh  DDP_standard electra 8 8 16

#######################################################################################################################
### overhead of context switch
## no heterogeneous 
## GEMM_ALGO not used actually

#patch -p0 < patches/no_context_switch.patch
cp /workspace/codes/EasyScale/easyscale/tasks/easyscale_task_no_cs.py /workspace/codes/EasyScale/easyscale/tasks/easyscale_task.py

EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=4 ./test.sh  EDDP_CPU_context_switch_disable vgg19 8 1 64
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=10 ./test.sh EDDP_CPU_context_switch_disable yolov3 8 1 8
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_context_switch_disable bert 8 1 16
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_context_switch_disable swintransformer 8 1 64

EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=7 ./test.sh  EDDP_CPU_context_switch_disable resnet50 8 1 128
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=7 ./test.sh  EDDP_CPU_context_switch_disable shufflenetv2 8 1 256
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_context_switch_disable ncf 8 1 16384
EASYSCALE_DETERMINISM_LEVEL=1 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_context_switch_disable electra 8 1 16

#patch -RE -p0 < patches/no_context_switch.patch
cp /workspace/codes/EasyScale/easyscale/tasks/easyscale_task_ori.py /workspace/codes/EasyScale/easyscale/tasks/easyscale_task.py


#   #######################################################################################################################
### overhead of gradient worker
## no heterogeneous 
## GEMM_ALGO not used actually

#patch -p0 < patches/no_gradient_worker.patch
cp /workspace/codes/EasyScale/easyscale/tasks/easyscale_task_no_gw.py /workspace/codes/EasyScale/easyscale/tasks/easyscale_task.py
cp /workspace/codes/EasyScale/easyscale/trainer_no_gw.py /workspace/codes/EasyScale/easyscale/trainer.py

HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=4 ./test.sh  EDDP_CPU_gradient_worker_disable vgg19 8 1 64
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=10 ./test.sh EDDP_CPU_gradient_worker_disable yolov3 8 1 8
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_gradient_worker_disable bert 8 1 16
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_gradient_worker_disable swintransformer 8 1 64

HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=7 ./test.sh  EDDP_CPU_gradient_worker_disable resnet50 8 1 128
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=7 ./test.sh  EDDP_CPU_gradient_worker_disable shufflenetv2 8 1 256
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_gradient_worker_disable ncf 8 1 16384
HETEROGENEOUS_DETERMINISTIC=0 MICRO_BATCH_LOG=1 GEMM_ALGO_ID=11 ./test.sh EDDP_CPU_gradient_worker_disable electra 8 1 16

#patch -RE -p0 < patches/no_gradient_worker.patch
cp /workspace/codes/EasyScale/easyscale/tasks/easyscale_task_ori.py /workspace/codes/EasyScale/easyscale/tasks/easyscale_task.py
cp /workspace/codes/EasyScale/easyscale/trainer_ori.py /workspace/codes/EasyScale/easyscale/trainer.py
