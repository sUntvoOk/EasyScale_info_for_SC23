#!/bin/bash
set -x
docker exec    easyscale bash /workspace/accuracy-consistency/run_eddp.sh resnet50 0 172.30.81.109 0 2 30 64 2 1 0 CPU 1 4  
docker exec -d   easyscale bash /workspace/accuracy-consistency/run_eddp.sh resnet50 1 172.30.81.109 1 2 30 64 2 1 1 CPU 1 4 

