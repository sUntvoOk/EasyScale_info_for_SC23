#!/bin/bash

script='./run_exp.sh'

export EXPERIMENT_STEPS=10 
export PYTORCH_GPU_VMEM=false 
export DATALOADER_WORKERS=0

# "GEMM_ALGO_ID=24" represents using the default algo of pytorch

####################

max_workers=16

model="shufflenetv2"
for ((i=1;i<`expr 1 \+ ${max_workers}`;i++))
do
ARTIFACT_MODE='PROFILING' \
HETEROGENEOUS_DETERMINISTIC=0 GEMM_ALGO_ID=24 MICRO_BATCH_LOG=1 ${script} EDDP_CPU ${model} imagenet float32 $i 1 512
pkill python
sleep 1
done
  
for ((i=1;i<`expr 1 \+ ${max_workers}`;i++))
do
ARTIFACT_MODE='PROFILING' \
HETEROGENEOUS_DETERMINISTIC=0 GEMM_ALGO_ID=24 ${script} DDP ${model} imagenet float32 $i $i 512
pkill python
sleep 1
done


model="resnet50"
for ((i=1;i<`expr 1 \+ ${max_workers}`;i++))
do
ARTIFACT_MODE='PROFILING' \
HETEROGENEOUS_DETERMINISTIC=0 GEMM_ALGO_ID=24 MICRO_BATCH_LOG=1 ${script} EDDP_CPU ${model} imagenet float32 $i 1 32
pkill python
sleep 1
done
  
for ((i=1;i<`expr 1 \+ ${max_workers}`;i++))
do
ARTIFACT_MODE='PROFILING' \
HETEROGENEOUS_DETERMINISTIC=0 GEMM_ALGO_ID=24 ${script} DDP ${model} imagenet float32 $i $i 32
pkill python
sleep 1
done
