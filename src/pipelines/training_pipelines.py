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
"""Training pipelines."""

import json
import time
import os

from . import components
from . import config
from kfp.v2 import dsl

GKE_ACCELERATOR_KEY = 'cloud.google.com/gke-accelerator'


@dsl.pipeline(
    name=config.TRAINING_PIPELINE_NAME,
    pipeline_root=config.TRAINING_PIPELINE_ROOT
)
def training_pipeline(
    bucket_name: str,
    train_prefix: str,
    valid_prefix: str,
    num_output_files_train: int,
    num_output_files_valid: int,
    output_path_defined_dir: str,
    output_path_analyzed_dir: str,
    output_path_transformed_dir: str,
    shuffle: str,
    version: str,
):
    '''
    
    Pipeline to preprocess parquet files in GCS.
    
    '''
    
    # =========================================================
    # TODO: extract from BQ to parquet 
    # =========================================================
    
    
    # =========================================================
    #             Convert from parquet to def 
    # =========================================================
    BUCKET_NAME = 'spotify-builtin-2t' # 'spotify-merlin-v1' # TODO: parameterize
    
    parquet_to_def_train = (
        components.convert_parquet_op(
            bucket_name=BUCKET_NAME,
            data_prefix=train_prefix,
            # data_dir_pattern=train_pattern,
            split='train',
            num_output_files=num_output_files_train,
            n_workers=int(config.GPU_LIMIT),
            shuffle=shuffle,
            output_path_defined_dir=output_path_defined_dir,
        )
    )
    parquet_to_def_train.set_display_name('Convert training split')
    parquet_to_def_train.set_cpu_limit(config.CPU_LIMIT)
    parquet_to_def_train.set_memory_limit(config.MEMORY_LIMIT)
    parquet_to_def_train.set_gpu_limit(config.GPU_LIMIT)
    parquet_to_def_train.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    parquet_to_def_train.set_caching_options(enable_caching=True)
    
    # === Convert eval dataset from CSV to Parquet
    parquet_to_def_valid = (
        components.convert_parquet_op(
            bucket_name=BUCKET_NAME,
            data_prefix=valid_prefix,
            # data_dir_pattern=valid_pattern,
            split='valid',
            num_output_files=num_output_files_valid,
            n_workers=int(config.GPU_LIMIT),
            shuffle=shuffle,
            output_path_defined_dir=output_path_defined_dir,
        )
    )
    parquet_to_def_valid.set_display_name('Convert validation split')
    parquet_to_def_valid.set_cpu_limit(config.CPU_LIMIT)
    parquet_to_def_valid.set_memory_limit(config.MEMORY_LIMIT)
    parquet_to_def_valid.set_gpu_limit(config.GPU_LIMIT)
    parquet_to_def_valid.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    parquet_to_def_valid.set_caching_options(enable_caching=True)

    # =========================================================
    # Analyse train dataset 
    # =========================================================
    
    # === Analyze train data split
    analyze_dataset = (
        components.analyze_dataset_op(
            # parquet_dataset=config.TRAIN_DIR_PARQUET,
            parquet_dataset=parquet_to_def_train.outputs['output_dataset'],
            n_workers=int(config.GPU_LIMIT),
            output_path_defined_dir=output_path_defined_dir,
            output_path_analyzed_dir=output_path_analyzed_dir
        )
    )
    analyze_dataset.set_display_name('Analyze Dataset')
    analyze_dataset.set_cpu_limit(config.CPU_LIMIT)
    analyze_dataset.set_memory_limit(config.MEMORY_LIMIT)
    analyze_dataset.set_gpu_limit(config.GPU_LIMIT)
    analyze_dataset.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    analyze_dataset.set_caching_options(enable_caching=True)

    # =========================================================
    # Transform train split 
    # =========================================================

    # === Transform train data split
    transform_train = (
        components.transform_dataset_op(
            workflow=analyze_dataset.outputs['workflow'],
            split='train',
            # parquet_dataset=config.TRAIN_DIR_PARQUET,
            parquet_dataset=parquet_to_def_train.outputs['output_dataset'],
            output_path_defined_dir=output_path_defined_dir,
            output_path_transformed_dir=f'{output_path_transformed_dir}',
            output_path_analyzed_dir=output_path_analyzed_dir,
            num_output_files=num_output_files_train,
            n_workers=int(config.GPU_LIMIT),
            version=version,
        )
    )
    transform_train.set_display_name('Transform train split')
    transform_train.set_cpu_limit(config.CPU_LIMIT)
    transform_train.set_memory_limit(config.MEMORY_LIMIT)
    transform_train.set_gpu_limit(config.GPU_LIMIT)
    transform_train.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    transform_train.set_caching_options(enable_caching=True)

    # =========================================================
    #     Transform valid split
    # =========================================================
    
    transform_valid = (
        components.transform_dataset_op(
            workflow=analyze_dataset.outputs['workflow'],
            split='valid',
            parquet_dataset=parquet_to_def_valid.outputs['output_dataset'],
            output_path_defined_dir=output_path_defined_dir,
            output_path_transformed_dir=f'{output_path_transformed_dir}',
            output_path_analyzed_dir=output_path_analyzed_dir,
            num_output_files=num_output_files_valid,
            n_workers=int(config.GPU_LIMIT),
            version=version,
        )
    )
    transform_valid.set_display_name('Transform valid split')
    transform_valid.set_cpu_limit(config.CPU_LIMIT)
    transform_valid.set_memory_limit(config.MEMORY_LIMIT)
    transform_valid.set_gpu_limit(config.GPU_LIMIT)
    transform_valid.add_node_selector_constraint(GKE_ACCELERATOR_KEY, config.GPU_TYPE)
    transform_valid.set_caching_options(enable_caching=True)