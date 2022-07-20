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

  # =============================================
  #            convert_to_parquet_op
  # =============================================
@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def convert_parquet_op(
    output_dataset: Output[Dataset],
    data_paths: list,
    split: str,
    num_output_files: int,
    n_workers: int,
    shuffle: Optional[str] = None,
    recursive: Optional[bool] = False,
    device_limit_frac: Optional[float] = 0.6,
    device_pool_frac: Optional[float] = 0.9,
    frac_size: Optional[float] = 0.10,
    memory_limit: Optional[int] = 100_000_000_000
):
  r"""Component to create NVTabular definition.
  Args:
    output_dataset: Output[Dataset]
      Output metadata with references to the converted CSV files in GCS
      and the split name.The path to the files are in GCS fuse format:
      /gcs/<bucket name>/path/to/file
    data_paths: list
      List of paths to folders or files on GCS.
      For recursive folder search, set the recursive variable to True:
        'gs://<bucket_name>/<subfolder1>/<subfolder>/' or
        'gs://<bucket_name>/<subfolder1>/<subfolder>/flat_file.csv' or
        a combination of both.
    split: str
      Split name of the dataset. Example: train or valid
    shuffle: str
      How to shuffle the converted CSV, default to None. Options:
        PER_PARTITION
        PER_WORKER
        FULL
    recursive: bool
      Recursivelly search for files in path.
    device_limit_frac: Optional[float] = 0.6
    device_pool_frac: Optional[float] = 0.9
    frac_size: Optional[float] = 0.10
    memory_limit: Optional[int] = 100_000_000_000
  """
  import os
  import logging

  from task import (
      create_cluster,
      create_parquet_dataset_definition,
      convert_definition_to_parquet,
      # get_criteo_col_dtypes,
  )

  logging.info('Base path in %s', output_dataset.path)

  # Write metadata
  output_dataset.metadata['split'] = split

  logging.info('Creating cluster')
  create_cluster(
    n_workers=n_workers,
    device_limit_frac=device_limit_frac,
    device_pool_frac=device_pool_frac,
    memory_limit=memory_limit
  )
  
  logging.info(f'Creating dataset definition from: {data_paths}')
  dataset = create_parquet_dataset_definition(
    data_paths=data_paths,
    recursive=recursive,
    # col_dtypes=get_criteo_col_dtypes(),
    frac_size=frac_size
  )
  
  logging.info(f'Converting definition to Parquet; {output_dataset.uri}')
  convert_definition_to_parquet(
    output_path=output_dataset.uri,
    dataset=dataset,
    output_files=num_output_files,
    shuffle=shuffle
  )

  # =============================================
  #            analyze_dataset_op
  # =============================================
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
    parquet_dataset: List of strings
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
      create_nvt_workflow,
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
      path_or_source=parquet_dataset,
      engine='parquet',
      part_mem_fraction=frac_size
  )

  logging.info('Creating Workflow')
  # Create Workflow
  nvt_workflow = create_nvt_workflow()

  logging.info('Analyzing dataset')
  nvt_workflow = nvt_workflow.fit(dataset)

  logging.info('Saving Workflow')
  nvt_workflow.save(workflow.path)
    
@dsl.component(
  base_image=config.NVT_IMAGE_URI,
  install_kfp_package=False
)
def transform_dataset_op(
    workflow: Input[Artifact],
    parquet_dataset: Input[Dataset],
    transformed_dataset: Output[Dataset],
    # output_dir: str,
    split: str,
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

  transformed_dataset.metadata['split'] = split
  
  logging.info('Creating cluster')
  create_cluster(
    n_workers=n_workers,
    device_limit_frac=device_limit_frac,
    device_pool_frac=device_pool_frac,
    memory_limit=memory_limit
  )

  logging.info(f'Creating Parquet dataset:{parquet_dataset.uri}')
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
  logging.info(f'transformed_dataset.path: {transformed_dataset.path}.')
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