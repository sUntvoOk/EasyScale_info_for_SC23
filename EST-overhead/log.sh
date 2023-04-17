#!/bin/bash

set -x
ulimit -c 0

run_test(){
    mode=$1
    model=$2
    mp=$3
    pod=$4
    bs=$5

    log_file=${log_dir}/mp${mp}_pod${pod}

    cd $model_dir

    sh kill.sh
    nohup python nvml.py ${log_file} &
    
    if [ ${mode::3} == 'DDP' ]; then
        gpus=""
        for ((i=0;i<${mp};i++))
        do
            if [ ${i} != $[${mp}-1] ]; then
                gpus=${gpus}${i}","
            else
                gpus=${gpus}${i}
            fi
        done
        CUDA_VISIBLE_DEVICES=${gpus} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 python -u ${model_dir}/ddp.py -n 1 -g ${mp} -nr 0 -e 10 --batch-size ${bs} -m ${model} 2>&1 | tee ${log_file}
    elif [ ${mode::8} == 'EDDP_GPU' ]; then
        for ((i=0;i<$[$pod-1];i++))
        do
            AIMASTER_ADDR=127.0.0.1:13245 CUDA_VISIBLE_DEVICES=${i} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 RANK=${i} WORLD_SIZE=${pod} python ${model_dir}/eddp.py -e 10 --batch-size ${bs} -m ${model} -mp ${mp} -grad-pos GPU &
        done
        AIMASTER_ADDR=127.0.0.1:13245 CUDA_VISIBLE_DEVICES=${i} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 RANK=${i} WORLD_SIZE=${pod} python ${model_dir}/eddp.py -e 10 --batch-size ${bs} -m ${model} -mp ${mp} -grad-pos GPU 2>&1 | tee ${log_file}
    elif [ ${mode::8} == 'EDDP_CPU' ]; then
        for ((i=0;i<$[$pod-1];i++))
        do
            AIMASTER_ADDR=127.0.0.1:13245 CUDA_VISIBLE_DEVICES=${i} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 RANK=${i} WORLD_SIZE=${pod} python ${model_dir}/eddp.py -e 10 --batch-size ${bs} -m ${model} -mp ${mp} -grad-pos CPU &
        done
        AIMASTER_ADDR=127.0.0.1:13245 CUDA_VISIBLE_DEVICES=${i} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 RANK=${i} WORLD_SIZE=${pod} python ${model_dir}/eddp.py -e 10 --batch-size ${bs} -m ${model} -mp ${mp} -grad-pos CPU 2>&1 | tee ${log_file}
    fi
    
    sh kill.sh
}

gpu=$1
mode=$2
model=$3
dataset=$4
float=$5
mp=$6
pod=$7
bs=$8

examples_dir=/workspace/codes/examples

if [ $model == "resnet50" ] || [ $model == "vgg19" ] || [ $model == "shufflenetv2" ]; then
    model_dir=${examples_dir}/imagenet
elif [ $model == "bert" ] || [ $model == "electra" ]; then
    model_dir=${examples_dir}/$model/src
else
    model_dir=${examples_dir}/$model
fi


#exp_dir=/workspace/reproduce-figures/figure10
SHELL_FOLDER=$(dirname $(readlink -f "$0"))

log_dir=${SHELL_FOLDER}/log/${model}/${gpu}/${mode}/${model}_${dataset}_${float}_mp${mp}_bs${bs}

if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi

echo !!!MODE ${PYTORCH_GPU_VMEM} ${ARTIFACT_MODE} ${EXPERIMENT_STEPS}

run_test ${mode} ${model} ${mp} ${pod} ${bs}
pkill python
sleep 1
