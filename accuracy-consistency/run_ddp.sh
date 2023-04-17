#!/bin/bash

examples_dir="/workspace/codes/examples/"
log_dir="/workspace/accuracy-consistency/"

heterogeneous_determinism=$4

model=$1
batch_size=$2
mp=$3

run_ddp_cmd()
{
    if [ $model == 'shufflenetv2' ] || [ $model == 'resnet50' ] || [ $model == 'vgg19' ]; then
        model_dir=${examples_dir}/imagenet
    elif [ $model == "bert" ] || [ $model == "electra" ]; then
        model_dir=${examples_dir}/$model/src
    else
        model_dir=${examples_dir}/$model
    fi

    if [ $heterogeneous_determinism == '1' ]; then
        log_dir=${log_dir}/logs_hete/
    elif [ $heterogeneous_determinism == '0' ]; then
        log_dir=${log_dir}/logs_homo/
    fi

    cd $model_dir

    if [ ! -d ${log_dir}${model} ]; then
        mkdir -p ${log_dir}${model}
    fi

    gpus=""
    for ((i=0;i<${mp};i++))
    do
        if [ ${i} != $[${mp}-1] ]; then
            gpus=${gpus}${i}","
        else
            gpus=${gpus}${i}
        fi
    done
    log_file=${log_dir}${model}/ddp.txt
    CUDA_VISIBLE_DEVICES=${gpus} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 python -u ddp.py -n 1 -g ${mp} -nr 0 -e 30 --batch-size ${batch_size} -m ${model} 2>&1 | tee ${log_file}

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

ARTIFACT_MODE='VALIDATING_ACCURACY' \
EXPERIMENT_STEPS=300 \
EXPERIMENT_STEPS_EPOCH=6 \
HETEROGENEOUS_DETERMINISTIC=${heterogeneous_determinism} PYTORCH_GPU_VMEM=false GEMM_ALGO_ID=$gemm_algo_id run_ddp_cmd
