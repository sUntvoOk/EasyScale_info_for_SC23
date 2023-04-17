# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class JobInfo(object):
    def __init__(self, creation_timestamp, placement
                 max_parallelism, proposals):
        assert max_parallelism > 0
        self.placement = placement
        self.creation_timestamp = creation_timestamp
        self.max_parallelism = max_parallelism
        self.proposals = proposals

class NodeInfo(object):
    def __init__(self, resources, preemptible):
        """
        Args:
            resources (dict): Available resources (eg. GPUs) on this node.
            preemptible (bool): Whether this node is pre-emptible.
        """
        #self.gpu_type = gpu_type
        # resources={"nvidia.com/gpu": self.num_gpus, "gpu_type": gpu}
        self.resources = resources
        self.preemptible = preemptible
