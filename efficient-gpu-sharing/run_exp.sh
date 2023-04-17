#!/bin/bash
set -x
ulimit -c 0

examples_dir="/workspace/codes/examples/"

run_test(){
    version=$1
    model=$2
    mp=$3
    pod=$4
    bs=$5

    if [ $model == 'shufflenetv2' ] || [ $model == 'resnet50' ] || [ $model == 'vgg19' ]; then
        model_dir=${examples_dir}/imagenet
    elif [ $model == "bert" ] || [ $model == "electra" ]; then
        model_dir=${examples_dir}/$model/src
    else
        model_dir=${examples_dir}/$model
    fi

    if [ $version == "DDP" ]; then
        log_file=${log_dir}/mp${mp}
    else
        log_file=${log_dir}/mp${mp}_pod${pod}
    fi

    cd $model_dir

    sh kill.sh
    nohup python nvml.py ${log_file} &

    case $version in
        DDP)
            gpus=""
            for ((i=0;i<${mp};i++))
            do
                if [ ${i} != $[${mp}-1] ]; then
                    gpus=${gpus}${i}","
                else
                    gpus=${gpus}${i}
                fi
            done
            ## CUDA_VISIBLE_DEVICES=${gpus} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 python -u ddp.py -n 1 -g ${mp} -nr 0 -e 100 --batch-size ${bs} -m ${model} 2>&1 | tee ${log_file}
	    ## TAG CHANGED
            #CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 python -u ${model_dir}/ddp.py -n 1 -g ${mp} -nr 0 -e 100 --batch-size ${bs} -m ${model} 2>&1 | tee ${log_file}
            CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 python -u ${model_dir}/ddp_single_gpu_multiple_worker.py -n 1 -g ${mp} -nr 0 -e 100 --batch-size ${bs} -m ${model} 2>&1 | tee ${log_file}
            ;;
        EDDP_GPU)
            for ((i=0;i<$[$pod-1];i++))
            do
                AIMASTER_ADDR=127.0.0.1:13245 CUDA_VISIBLE_DEVICES=${i} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 RANK=${i} WORLD_SIZE=${pod} python ${model_dir}/eddp.py -e 100 --batch-size ${bs} -m ${model} -mp ${mp} -grad-pos GPU &
            done
            AIMASTER_ADDR=127.0.0.1:13245 CUDA_VISIBLE_DEVICES=${i} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 RANK=${i} WORLD_SIZE=${pod} python ${model_dir}/eddp.py -e 100 --batch-size ${bs} -m ${model} -mp ${mp} -grad-pos GPU 2>&1 | tee ${log_file}
            ;;
        EDDP_CPU)
            for ((i=0;i<$[$pod-1];i++))
            do
                AIMASTER_ADDR=127.0.0.1:13245 CUDA_VISIBLE_DEVICES=${i} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 RANK=${i} WORLD_SIZE=${pod} python ${model_dir}/eddp.py -e 100 --batch-size ${bs} -m ${model} -mp ${mp} -grad-pos CPU &
            done
            AIMASTER_ADDR=127.0.0.1:13245 CUDA_VISIBLE_DEVICES=${i} MASTER_ADDR=127.0.0.1 MASTER_PORT=9561 RANK=${i} WORLD_SIZE=${pod} python ${model_dir}/eddp.py -e 100 --batch-size ${bs} -m ${model} -mp ${mp} -grad-pos CPU 2>&1 | tee ${log_file}
            ;;
    esac
    
    sh kill.sh
}

# 参数1：version(DDP, EDDP_GPU, EDDP_GPU)
# 参数2：model(resnet50/resnet18/vgg19/mobilenetv2)
# 参数3：dataset(cifar10/imagenet)
# 参数4：float32/float64，需要改ddp和easyscale的set_seed
# 参数5：mp
# 参数6：pod
# 参数7：bs

# ./run.sh DDP resnet50 imagenet float32 8 8 64
# ./run.sh EDDP_GPU resnet50 imagenet float32 8 8 64

version=$1
model=$2
dataset=$3
float=$4
mp=$5
pod=$6
bs=$7


SHELL_FOLDER=$(dirname $(readlink -f "$0"))

if [ $version == "DDP" ]; then
    log_dir=${SHELL_FOLDER}/log/${version}/${model}_${dataset}_${float}_bs${bs}
    
else
    log_dir=${SHELL_FOLDER}/log/${version}/${model}_${dataset}_${float}_mp${mp}_bs${bs}
fi

if [ ! -d $log_dir ]; then
    mkdir -p $log_dir
fi


run_test ${version} ${model} ${mp} ${pod} ${bs}
