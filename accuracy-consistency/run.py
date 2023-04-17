# -*- coding: utf-8 -*- 

import math
import os, sys, subprocess
import numpy as np
import shutil
import pickle
import json
import time

sys.path.append('../scripts')
import variables
#variables.set_value('_LOG_DIR', 'logs')
#variables.set_value('_EXEC_BASE', 'bash /workspace/accuracy-consistency/run_eddp.sh')
variables.set_value('_EXEC_BASE', 'bash /workspace/accuracy-consistency/run_eddp_workers.sh')
from step1_run_proposals import Profiling, save_log_file

## 3 stages
validating_configs = {
  'stage_0': Profiling.generate_a_proposal(placement={'v100': 4, 'p100': 0, 't4': 0}, threads={'v100': 1, 'p100': 0, 't4': 0}, processes={'v100': 1, 'p100': 1, 't4': 1}, mp=4), 
  'stage_1': Profiling.generate_a_proposal(placement={'v100': 2, 'p100': 0, 't4': 0}, threads={'v100': 2, 'p100': 0, 't4': 0}, processes={'v100': 1, 'p100': 1, 't4': 1}, mp=4), 
  'stage_2': Profiling.generate_a_proposal(placement={'v100': 1, 'p100': 2, 't4': 0}, threads={'v100': 2, 'p100': 1, 't4': 0}, processes={'v100': 1, 'p100': 1, 't4': 1}, mp=4), 

  #'test': Profiling.generate_a_proposal(placement={'v100': 2, 'p100': 0, 't4': 0}, threads={'v100': 1, 'p100': 0, 't4': 0}, processes={'v100': 1, 'p100': 0, 't4': 0}, mp=2), 
}

# Dataloader workers milestone
# bert, electra
# resnet50, swintransformer

batch_sizes = {
  #'ncf':               4096,
  #'yolov3':            4,
  #'bert':              8,
  #'electra':           8,
  'resnet50':          64,
  #'swintransformer':   32,
  #'shufflenetv2':      128,
  'vgg19':             64,
}


def run_ddp(model, mp=4, heterogeneous_determinism=1):
  bs = batch_sizes[model]
  cmd = "bash /workspace/accuracy-consistency/run_ddp.sh {} {} {} {}".format(model, bs, mp, heterogeneous_determinism)
  ret = "ssh root@v0 \" docker exec easyscale {} \" ".format(cmd)
  print(ret)
  process = subprocess.Popen(ret, shell=True, bufsize=1)
  process.wait()

# DDP baseline
#for application_name in batch_sizes:
#  run_ddp(application_name, mp=4, heterogeneous_determinism=0)
#  run_ddp(application_name, mp=4, heterogeneous_determinism=1)
#exit(0)


def rm_ckpts():
  cmd = 'bash /workspace/accuracy-consistency/rm_ckpt.sh {}'.format(application_name)
  ret = "ssh root@v0 \" docker exec easyscale {} \" ".format(cmd)
  process = subprocess.Popen(ret, shell=True, bufsize=1)
  process.wait()

def kill_all():
  process = subprocess.Popen("bash ../scripts/0_setup_nodes.sh kill", shell=True, stdout=devNull, stderr=devNull)
  process.wait()

# EasyScale
profiling = Profiling()
for application_name in batch_sizes:
  devNull = open(os.devnull, 'w')
  # O0, O1, O0+O2, O1+O2
  for determinism_level in [2,12,0]:
    for dataloader_workers in [4]:
      rm_ckpts()
      for stage in validating_configs:
      #for stage in ['stage_0']:
        kill_all()
        profiling.run_a_proposal( application_name=application_name, 
                                  proposal=validating_configs[stage], 
                                  epoch=30, 
                                  batch_size=batch_sizes[application_name],
                                  determinism_level=determinism_level,
                                  dataloader_workers=dataloader_workers)

        if determinism_level == 0:
          path = os.path.join('./logs_O0', application_name)
        elif determinism_level == 2:
          path = os.path.join('./logs_O02', application_name)
        elif determinism_level == 1:
          # hack
          path = os.path.join('./logs_O1', application_name)
          #path = os.path.join('./logs_workers', application_name)
        if determinism_level == 12:
          path = os.path.join('./logs_O12', application_name)
        # hack
        #file_name = str(stage) + '.txt'
        file_name = "{}_{}.txt".format(stage, dataloader_workers)
        log_path = save_log_file(path, file_name)

        time.sleep(5)

        



## 5 stages
# validating_configs = {
#   'stage_0': Profiling.generate_a_proposal(placement={'v100': 1, 'p100': 0, 't4': 0}, threads={'v100': 4, 'p100': 0, 't4': 0}, mp=4), 
#   'stage_1': Profiling.generate_a_proposal(placement={'v100': 2, 'p100': 0, 't4': 0}, threads={'v100': 2, 'p100': 0, 't4': 0}, mp=4), 
#   'stage_2': Profiling.generate_a_proposal(placement={'v100': 4, 'p100': 0, 't4': 0}, threads={'v100': 1, 'p100': 0, 't4': 0}, mp=4), 
#   'stage_3': Profiling.generate_a_proposal(placement={'v100': 1, 'p100': 2, 't4': 0}, threads={'v100': 2, 'p100': 1, 't4': 0}, mp=4), 
#   'stage_4': Profiling.generate_a_proposal(placement={'v100': 1, 'p100': 0, 't4': 2}, threads={'v100': 2, 'p100': 0, 't4': 1}, mp=4), 
#   'stage_5': Profiling.generate_a_proposal(placement={'v100': 0, 'p100': 2, 't4': 2}, threads={'v100': 0, 'p100': 2, 't4': 2}, mp=4), 
#   
#   #'debug': Profiling.generate_a_proposal(placement={'v100': 1, 'p100': 0, 't4': 0}, threads={'v100': 1, 'p100': 0, 't4': 0}, mp=1) 
# }
