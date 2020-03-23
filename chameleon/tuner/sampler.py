# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument, no-self-use, invalid-name
""" Base class of samplers
    [*] Chameleon: Adaptive Code Optimization for
        Expedited Deep Neural Network Compilation
        Byung Hoon Ahn, Prannoy Pilligundla, Amir Yazdanbakhsh, Hadi Esmaeilzadeh
        https://openreview.net/forum?id=rygG4AVFvH
"""
import logging

import numpy as np

from ..env import GLOBAL_SCOPE

logger = logging.getLogger('autotvm')

class Sampler(object):
    """Base class for samplers

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task
    plan_size: int
        The number of samples the sampler receives
    """

    def __init__(self, task, plan_size=64):
        # space
        self.task = task
        self.space = task.config_space
        self.dims = [len(x) for x in self.space.space_map.values()]

        self.plan_size = plan_size

    def sample(self, xs):
        """Sample

        Parameters
        ----------
        xs: Array of int
            The indices of configs from the optimizer
        """
        return xs

