#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: 
#         Gengzhi Yang <genzyang17@gmail.com>
#         Hong-Zhou Ye <hzyechem@gmail.com>
#


import numpy as np

from .aft import AFTDF
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools


class AFTDF_STC(AFTDF):

    def range_coulomb(self, omega):
