#!/bin/bash

examples_dir="/workspace/codes/examples/"

determinism_level=${12}
dataloader_workers=${13}

model=$1
local_rank=$2
master_addr=$3
global_rank=$4
world_size=$5
epoch=$6
batch_size=$7
mp=$8
ap=$9
start_rank=${10}
grad_pos=${11}

run_eddp_cmd()
{
    model=$1

    if [ $model == 'shufflenetv2' ] || [ $model == 'resnet50' ] || [ $model == 'vgg19' ]; then
        model_dir=${examples_dir}/imagenet
    elif [ $model == "bert" ] || [ $model == "electra" ]; then
        model_dir=${examples_dir}/$model/src
    else
        model_dir=${examples_dir}/$model
    fi

    cd $model_dir

    if [ ! -d "./ckpt" ]; then
        mkdir ./ckpt
    fi

    echo OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${local_rank} MASTER_ADDR=${master_addr} MASTER_PORT=9561 \
        RANK=${global_rank} WORLD_SIZE=${world_size} \
        python eddp.py -e ${epoch} --batch-size ${batch_size} -m ${model} -mp ${mp} -ap ${ap} -start-rank ${start_rank} -grad-pos ${grad_pos}
    
    TORCH_DISTRIBUTED_DEBUG=DETAIL OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${local_rank} MASTER_ADDR=${master_addr} MASTER_PORT=9561 \
        RANK=${global_rank} WORLD_SIZE=${world_size} \
        python eddp.py -e ${epoch} --batch-size ${batch_size} -m ${model} -mp ${mp} -ap ${ap} -start-rank ${start_rank} -grad-pos ${grad_pos}
}



if [ $model == 'shufflenetv2' ] || [ $model == 'resnet50' ]; then
    gemm_algo_id=7
elif [ $model == 'vgg19' ]; then
    gemm_algo_id=4
elif [ $model == 'ncf' ] || [ $model == 'bert' ] || [ $model == 'electra' ] || [ $model == 'swintransformer' ]; then
    gemm_algo_id=11
elif [ $model == 'yolov3' ]; then
    gemm_algo_id=10
else
    gemm_algo_id=2
fi

########################

#ARTIFACT_MODE='PROFILING' \
ARTIFACT_MODE='VALIDATING_ACCURACY' \
EXPERIMENT_STEPS=100 \
EXPERIMENT_STEPS_EPOCH=2 \
EASYSCALE_DETERMINISM_LEVEL=${determinism_level} \
DATALOADER_WORKERS=${dataloader_workers} \
PYTORCH_GPU_VMEM=false MICRO_BATCH_LOG=1 GEMM_ALGO_ID=$gemm_algo_id run_eddp_cmd $model

