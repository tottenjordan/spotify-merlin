# Copyright 2021 Google LLC
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
"""Preprocessing pipelines."""

from . import components
from . import config
from kfp.v2 import dsl

GKE_ACCELERATOR_KEY = 'cloud.google.com/gke-accelerator'

@dsl.pipeline(
    name=config.PREPROCESS_PARQUET_PIPELINE_NAME,
    pipeline_root=config.PREPROCESS_PARQUET_PIPELINE_ROOT
)
def preprocessing_parquet(
    train_paths: list,
    valid_paths: list,
    num_output_files_train: int,
    num_output_files_valid: int,
    shuffle: str
):
  # =================================
  # TODO: extract from BQ to parquet 
  # =================================

  # =================================
  # Analyse train dataset 
  # =================================
  # === Analyze train data split
  analyze_dataset = components.analyze_dataset_op(
      parquet_dataset=config.TRAIN_DIR_PARQUET,
      n_workers=int(config.GPU_LIMIT)
  )
  analyze_dataset.set_display_name('Analyze Dataset')#.set_caching_options(enable_caching=True)
  analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
  analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
  analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
  analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

  # ==================== Transform train and validation dataset =============

  # === Transform train data split
  transform_train = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      split='train',
      parquet_dataset=config.TRAIN_DIR_PARQUET,
      num_output_files=num_output_files_train,
      n_workers=int(config.GPU_LIMIT)
  )
  transform_train.set_display_name('Transform train split')
  transform_train.set_cpu_limit(config.CPU_LIMIT)
  transform_train.set_memory_limit(config.MEMORY_LIMIT)
  transform_train.set_gpu_limit(config.GPU_LIMIT)
  transform_train.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)

  # === Transform eval data split
  transform_valid = components.transform_dataset_op(
      workflow=analyze_dataset.outputs['workflow'],
      split='valid',
      parquet_dataset=config.VALID_DIR_PARQUET,
      num_output_files=num_output_files_valid,
      n_workers=int(config.GPU_LIMIT)
  )
  transform_valid.set_display_name('Transform valid split')
  transform_valid.set_cpu_limit(config.CPU_LIMIT)
  transform_valid.set_memory_limit(config.MEMORY_LIMIT)
  transform_valid.set_gpu_limit(config.GPU_LIMIT)
  transform_valid.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)