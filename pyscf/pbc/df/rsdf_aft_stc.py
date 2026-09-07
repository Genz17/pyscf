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

# use rsdf_stc for full Coulomb and aft_stc for LR Coulomb.
# for all-electron computation

import numpy as np

from .aft import AFTDF
from .rsdf_stc import RSGDF
from .stc_helper import get_coulG
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df import aft_jk, ft_ao
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import (
    is_zero, group_by_conj_pairs, kk_adapted_iter,
)
from pyscf.pbc.df.df_jk import (
    _format_dms, _ewald_exxdiv_for_G0,
)
from pyscf.pbc.df.aft_jk import (
    _update_vk_, _update_vk1_,
    _update_vk_dmf, _update_vk1_dmf,
    _mo_k2gamma, _gen_ft_kernel_fake_gamma,
    _update_vk_fake_gamma,
)
from pyscf.pbc.df.aft import _check_kpts
