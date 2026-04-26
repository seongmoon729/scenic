# Copyright 2026 The Scenic Authors.
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

"""raw_3ch_w_pt + nod_sony target: small-object dominant test set."""

from scenic.projects.baselines.centernet.configs.centernet2_CXT_LSJ_4x_raw_3ch_w_pt import get_config as get_base_config


def get_config():
  config = get_base_config()
  # sony test median object ~25x25 in 256² → ~100x100 in 1024² model space.
  config.dataset_configs.scale_range = (0.1, 1.5)
  return config
