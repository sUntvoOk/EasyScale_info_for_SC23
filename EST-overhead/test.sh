#!/bin/bash

set -x
ulimit -c 0

mode=$1
model=$2
mp=$3
pod=$4
bs=$5


case $model in
    resnet50)
        ./log.sh V100 $mode resnet50 imagenet float32 $mp $pod $bs
        ;;
    vgg19)
        ./log.sh V100 $mode vgg19 imagenet float32 $mp $pod $bs
        ;;
    ncf)
        ./log.sh V100 $mode ncf movielens float32 $mp $pod $bs
        ;;
    yolov3)
        ./log.sh V100 $mode yolov3 voc float32 $mp $pod $bs
        ;;
    bert)
        ./log.sh V100 $mode bert squad float32 $mp $pod $bs
        ;;
    deepspeech)
        ./log.sh V100 $mode deepspeech an4 float32 $mp $pod $bs
        ;;
    swintransformer)
        ./log.sh V100 $mode swintransformer imagenet float32 $mp $pod $bs
        ;;
    electra)
        ./log.sh V100 $mode electra squad float32 $mp $pod $bs
        ;;
    shufflenetv2)
        ./log.sh V100 $mode shufflenetv2 imagenet float32 $mp $pod $bs
        ;;
esac