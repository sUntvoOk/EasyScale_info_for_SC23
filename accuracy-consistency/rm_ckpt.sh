#!/bin/bash

examples_dir="/workspace/codes/examples/"

model=$1

run_rm_ckpt_cmd()
{
    if [ $model == 'shufflenetv2' ] || [ $model == 'resnet50' ] || [ $model == 'vgg19' ]; then
        model_dir=${examples_dir}/imagenet
    elif [ $model == "bert" ] || [ $model == "electra" ]; then
        model_dir=${examples_dir}/$model/src
    else
        model_dir=${examples_dir}/$model
    fi

    cd $model_dir

    if [ -d "./ckpt" ]; then
        rm -rf ./ckpt
        mkdir ./ckpt
    fi

}


run_rm_ckpt_cmd
