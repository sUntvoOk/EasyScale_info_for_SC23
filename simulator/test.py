from applications import APPLICATIONS
from simulator import Job

import pdb

print("#####################################")

app = APPLICATIONS['cifar10']

print(app.placements)

ret = app.get_throughput({"v100":[2], "p100":[1], "t4":[1]}, 32)

print(ret)

print("#####################################")

job = Job('cifar10-test', APPLICATIONS['cifar10'], '2017-10-09 07:01:55', 16, 64)

print(job.get_gpu_perf())
print(job.get_throughput({"v100":[2], "p100":[1], "t4":[1]}))

placement = {"v100":[2], "p100":[1], "t4":[1]}

for i, (k, v) in enumerate(placement.items()):
  print(k)
  print(v)

job.reallocate(placement)

job.step(60)
print(job.progress)

pdb.set_trace()