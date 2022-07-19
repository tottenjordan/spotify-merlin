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
"""KFP components."""

from typing import Optional
from . import config

from kfp.v2 import dsl
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Dataset
from kfp.v2.dsl import Input
from kfp.v2.dsl import Model
from kfp.v2.dsl import Output

@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def analyze_dataset_op(
    parquet_dataset: Input[Dataset],
    workflow: Output[Artifact],
    n_workers: int,
    device_limit_frac: Optional[float] = 0.6,
    device_pool_frac: Optional[float] = 0.9,
    frac_size: Optional[float] = 0.10,
    memory_limit: Optional[int] = 100_000_000_000
):
  """Component to generate statistics from the dataset.
  Args:
    parquet_dataset: Input[Dataset]
      Input metadata with references to the train and valid converted
      datasets in GCS and the split name.
    workflow: Output[Artifact]
      Output metadata with the path to the fitted workflow artifacts
      (statistics).
    device_limit_frac: Optional[float] = 0.6
    device_pool_frac: Optional[float] = 0.9
    frac_size: Optional[float] = 0.10
  """
  import logging
  import nvtabular as nvt
  
  from task import (
      create_cluster,
      create_criteo_nvt_workflow,
  )

  logging.basicConfig(level=logging.INFO)

  create_cluster(
    n_workers=n_workers,
    device_limit_frac=device_limit_frac,
    device_pool_frac=device_pool_frac,
    memory_limit=memory_limit
  )

  logging.info('Creating Parquet dataset')
  dataset = nvt.Dataset(
      path_or_source=parquet_dataset.uri,
      engine='parquet',
      part_mem_fraction=frac_size
  )

  logging.info('Creating Workflow')
  # Create Workflow
  criteo_workflow = create_criteo_nvt_workflow()

  logging.info('Analyzing dataset')
  criteo_workflow = criteo_workflow.fit(dataset)

  logging.info('Saving Workflow')
  criteo_workflow.save(workflow.path)
    
@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
    num_output_files: int,
    n_workers: int,
    shuffle: str = None,
    device_limit_frac: float = 0.6,
    device_pool_frac: float = 0.9,
    frac_size: float = 0.10,
    memory_limit: int = 100_000_000_000
):
  """Component to transform a dataset according to the workflow definitions.
  Args:
    workflow: Input[Artifact]
      Input metadata with the path to the fitted_workflow
    parquet_dataset: Input[Dataset]
      Location of the converted dataset in GCS and split name
    transformed_dataset: Output[Dataset]
      Split name of the transformed dataset.
    shuffle: str
      How to shuffle the converted CSV, default to None. Options:
        PER_PARTITION
        PER_WORKER
        FULL
    device_limit_frac: float = 0.6
    device_pool_frac: float = 0.9
    frac_size: float = 0.10
  """
  import os
  import logging
  import nvtabular as nvt
  from merlin.schema import Tags
  
  from task import (
    create_cluster,
    save_dataset,
  )

  logging.basicConfig(level=logging.INFO)

  transformed_dataset.metadata['split'] = \
    parquet_dataset.metadata['split']
  
  logging.info('Creating cluster')
  create_cluster(
    n_workers=n_workers,
    device_limit_frac=device_limit_frac,
    device_pool_frac=device_pool_frac,
    memory_limit=memory_limit
  )

  logging.info(f'Creating Parquet dataset: {parquet_dataset.uri}')
  dataset = nvt.Dataset(
      path_or_source=parquet_dataset.uri,
      engine='parquet',
      part_mem_fraction=frac_size
  )

  logging.info('Loading Workflow')
  nvt_workflow = nvt.Workflow.load(workflow.path)

  logging.info('Transforming Dataset')
  trans_dataset = nvt_workflow.transform(dataset)

  logging.info(f'Saving transformed dataset: {transformed_dataset.uri}')
  save_dataset(
    dataset=trans_dataset,
    output_path=transformed_dataset.uri,
    output_files=num_output_files,
    shuffle=shuffle
  )

  logging.info('Generating file list for training.')
  file_list = os.path.join(transformed_dataset.path, '_file_list.txt')

  new_lines = []
  with open(file_list, 'r') as fp:
    lines = fp.readlines()
    new_lines.append(lines[0])
    for line in lines[1:]:
      new_lines.append(line.replace('gs://', '/gcs/'))

  gcs_file_list = os.path.join(transformed_dataset.path, '_gcs_file_list.txt')
  with open(gcs_file_list, 'w') as fp:
    fp.writelines(new_lines)

  logging.info('Saving cardinalities')
  cols_schemas = nvt_workflow.output_schema.select_by_tag(Tags.CATEGORICAL)
  cols_names = cols_schemas.column_names

  cards = []
  for c in cols_names:
    col = cols_schemas.get(c)
    cards.append(col.properties['embedding_sizes']['cardinality'])

  transformed_dataset.metadata['cardinalities'] = cards