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

# pylint: disable=line-too-long
r"""Default configs for COCO detection using CenterNet.

"""
# pylint: enable=line-too-long

from scenic.projects.baselines.centernet.configs.centernet2_CXT_LSJ_4x import get_config as get_base_config


def get_config():
  """get config."""
  config = get_base_config()

  config.model.roi_num_classes = 3
  config.dataset_configs.train_data_path = 'rgb_od/nod_nikon'
  config.dataset_configs.test_data_path = 'rgb_od/nod_nikon'

  config.dataset_configs.cache = False
  config.dataset_configs.filter_labels = False

  config.num_training_steps = 50_000
  config.lr_configs.warmup_steps = 250
  config.lr_configs.base_learning_rate = 0.0001

  return config
